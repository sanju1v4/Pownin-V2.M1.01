import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StatusBar,
  Text,
  View,
} from 'react-native';
import {
  AudioModule,
  getRecordingPermissionsAsync,
  RecordingPresets,
  setAudioModeAsync,
  useAudioPlayer,
  useAudioPlayerStatus,
  useAudioRecorder,
} from 'expo-audio';

// ---------------------------------------------------------------------------
// The one thing that changes between demos.
// ---------------------------------------------------------------------------
const SERVER_URL = 'ws://10.6.66.30:8000/ws';

// Explicit capture config rather than HIGH_QUALITY as-shipped.
//
// HIGH_QUALITY records 44.1 kHz stereo, which the server immediately averages
// to mono and resamples to 16 kHz for DPDFNet -- so the extra rate and the
// second channel are thrown away after inflating the upload. Capturing at the
// pipeline's own rate removes a resample and shrinks the file ~4x.
//
// Note this does NOT change any near/far-field processing: RecordingOptions
// only describes the encoder (format, rate, bit rate). Nothing in it can
// select an AVAudioSession mode -- see the notes on far-field capture below.
//
// isMeteringEnabled is on so status.metering carries a real level; the preset
// leaves it unset, which reports undefined.
const RECORDING_OPTIONS = {
  ...RecordingPresets.HIGH_QUALITY,
  isMeteringEnabled: true,
  sampleRate: 16000,
  numberOfChannels: 1,
  bitRate: 64000,
  ios: {
    ...RecordingPresets.HIGH_QUALITY.ios,
    sampleRate: 16000,
  },
  android: {
    ...RecordingPresets.HIGH_QUALITY.android,
    sampleRate: 16000,
    numberOfChannels: 1,
    bitRate: 64000,
  },
};

const EXT = 'm4a';
const CHUNK_CHARS = 64 * 1024; // multiple of 4 -> every chunk decodes on its own
const MAX_RECORD_SECONDS = 600; // hard stop so a forgotten finger can't fill the disk

// expo-file-system moved to a class API; the legacy entry point is still the
// one that reliably reads a whole file to base64 across SDK versions.
const FS: any = (() => {
  try {
    return require('expo-file-system/legacy');
  } catch {
    return require('expo-file-system');
  }
})();

// Optional: keeps the screen alive through a three-minute hold. Guarded so a
// missing package can never crash the demo.
const KeepAwake: any = (() => {
  try {
    return require('expo-keep-awake');
  } catch {
    return null;
  }
})();

// ---------------------------------------------------------------------------
// Palette. Hardcoded, no theme system. Ember is the ENHANCED pane's alone.
// ---------------------------------------------------------------------------
const INK = '#1A1614';
const PAPER = '#FFFDF7';
const PAGE = '#FFF6E8';
const EMBER = '#FF6B35';
const GREEN = '#2BB673';
const RED = '#E24B4A';
const MUTED = '#F0E4CE';
const GREY_TEXT = '#5C554E';
const LABEL_GREY = '#8A8178';
const ENHANCED_FILL = '#FFF1E8';

const PANE_HEIGHT = 150;

type Phase = 'idle' | 'recording' | 'uploading' | 'processing' | 'done' | 'error';

const STAGE_LABELS: Record<string, string> = {
  decoded: 'Audio decoded',
  enhancing: 'Removing noise',
  transcribing: 'Transcribing enhanced audio',
};

function stageLabel(stage: string): string {
  if (STAGE_LABELS[stage]) return STAGE_LABELS[stage];
  return stage.charAt(0).toUpperCase() + stage.slice(1);
}

function clock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

type Settled = {
  uri: string | null;
  fileSize: number | null;
  mediaServicesDidReset: boolean;
  waitedMs: number;
};

/**
 * Wait for the recorder to actually come to rest after stop().
 *
 * stop() resolving only means the stop was accepted. An .m4a keeps its index
 * -- the moov atom -- at the very END of the file, written during finalisation.
 * Read the file before that flush lands and you get ftyp + mdat and nothing
 * else, which ffmpeg refuses outright with "moov atom not found".
 *
 * Do NOT gate on durationMillis here. A stopped recorder reports
 * durationMillis: 0 -- the counter is live-only -- so waiting for a non-zero
 * duration after stop() can never succeed and simply burns the whole timeout,
 * then reports a perfectly good 20 s recording as "0.0s". The duration is
 * captured during the hold instead (see maxDurationRef).
 *
 * The real finalisation signal is the file itself: poll until its size is
 * non-zero and unchanged across consecutive reads, which means the moov flush
 * has landed.
 */
async function settleRecorder(rec: any, timeoutMs = 3000): Promise<Settled> {
  const startedAt = Date.now();
  let sawReset = false;

  const readStatus = (): any => {
    try {
      const s = rec.getStatus?.() ?? null;
      if (s?.mediaServicesDidReset) sawReset = true;
      return s;
    } catch {
      return null;
    }
  };

  // 1. wait for the recorder to report it has stopped
  let status = readStatus();
  while (Date.now() - startedAt < timeoutMs) {
    const stillRecording = status?.isRecording ?? rec.isRecording ?? false;
    if (!stillRecording) break;
    await new Promise((r) => setTimeout(r, 50));
    status = readStatus();
  }

  const uri = rec.uri ?? status?.url ?? null;

  // 2. wait for the file to stop growing
  let size: number | null = null;
  let stableReads = 0;
  while (uri && Date.now() - startedAt < timeoutMs) {
    let current: number | null = null;
    try {
      const info = await FS.getInfoAsync(uri, { size: true });
      current = info?.exists ? (info.size ?? null) : null;
    } catch {
      current = null;
    }
    if (current != null && current > 0 && current === size) {
      if (++stableReads >= 2) break;
    } else {
      stableReads = 0;
    }
    size = current;
    await new Promise((r) => setTimeout(r, 50));
  }

  readStatus();
  return {
    uri,
    fileSize: size,
    mediaServicesDidReset: sawReset,
    waitedMs: Date.now() - startedAt,
  };
}

export default function App() {
  const recorder = useAudioRecorder(RECORDING_OPTIONS);

  const [phase, setPhase] = useState<Phase>('idle');
  const [granted, setGranted] = useState<boolean | null>(null);
  const [elapsed, setElapsed] = useState(0); // seconds, recording or processing
  const [stage, setStage] = useState('');
  const [pct, setPct] = useState(0);
  const [uploadPct, setUploadPct] = useState(0);
  const [rawNotes, setRawNotes] = useState('');
  const [enhancedNotes, setEnhancedNotes] = useState('');
  const [serverMs, setServerMs] = useState<number | null>(null);
  const [rawUrl, setRawUrl] = useState<string | null>(null);
  const [enhancedUrl, setEnhancedUrl] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [diag, setDiag] = useState('');

  // -- playback -------------------------------------------------------------
  // Created once with no source; the clip URLs only arrive with the result
  // frame, so the source is swapped in via replace().
  const rawPlayer = useAudioPlayer();
  const enhPlayer = useAudioPlayer();
  const rawPlayback = useAudioPlayerStatus(rawPlayer);
  const enhPlayback = useAudioPlayerStatus(enhPlayer);

  useEffect(() => {
    if (rawUrl) rawPlayer.replace({ uri: rawUrl });
  }, [rawUrl, rawPlayer]);

  useEffect(() => {
    if (enhancedUrl) enhPlayer.replace({ uri: enhancedUrl });
  }, [enhancedUrl, enhPlayer]);

  const togglePlay = useCallback(
    async (which: 'raw' | 'enh') => {
      const mine = which === 'raw' ? rawPlayer : enhPlayer;
      const other = which === 'raw' ? enhPlayer : rawPlayer;
      if (!(which === 'raw' ? rawUrl : enhancedUrl)) return;

      // The recording session leaves iOS in play-and-record, which routes
      // output to the earpiece and is far too quiet to demo. Hand the session
      // back to playback before making a sound; startRecording re-asserts
      // allowsRecording before the next hold.
      try {
        await setAudioModeAsync({ allowsRecording: false, playsInSilentMode: true });
      } catch (e: any) {
        // console directly: log() is declared further down, and pulling it
        // into this dependency array would be a use-before-declaration.
        console.log(`[rec] setAudioModeAsync for playback failed: ${String(e?.message ?? e)}`);
      }

      // only one pane plays at a time
      try {
        other.pause();
        await other.seekTo(0);
      } catch {
        /* the other pane may have no source yet */
      }

      if (mine.playing) {
        mine.pause();
        return;
      }
      // replay from the top if it ran to the end last time
      if (mine.duration > 0 && mine.currentTime >= mine.duration - 0.05) {
        await mine.seekTo(0);
      }
      mine.play();
    },
    [enhPlayer, enhancedUrl, rawPlayer, rawUrl],
  );

  const wsRef = useRef<WebSocket | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startedAtRef = useRef(0);
  const busyRef = useRef(false); // guards against double onPressIn
  const phaseRef = useRef<Phase>('idle');
  phaseRef.current = phase;

  // -- recorder diagnostics -------------------------------------------------
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const maxDurationRef = useRef(0); // highest durationMillis seen during a hold
  const meterSeenRef = useRef(false); // did metering ever report a finite level?

  const log = useCallback((msg: string) => {
    console.log(`[rec] ${msg}`);
    setDiag(msg);
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = null;
  }, []);

  /** Poll getStatus() through the hold so a dead mic is visible immediately. */
  const startPolling = useCallback(() => {
    stopPolling();
    maxDurationRef.current = 0;
    meterSeenRef.current = false;
    pollRef.current = setInterval(() => {
      let s: any = null;
      try {
        s = recorder.getStatus?.() ?? null;
      } catch (e: any) {
        log(`getStatus() threw: ${String(e?.message ?? e)}`);
        return;
      }
      const dur = typeof s?.durationMillis === 'number' ? s.durationMillis : -1;
      if (dur > maxDurationRef.current) maxDurationRef.current = dur;
      if (typeof s?.metering === 'number' && Number.isFinite(s.metering)) {
        meterSeenRef.current = true;
      }
      log(
        `dur=${dur}ms t=${(recorder.currentTime ?? -1).toFixed(2)}s ` +
          `rec=${s?.isRecording} can=${s?.canRecord} ` +
          `met=${typeof s?.metering === 'number' ? s.metering.toFixed(1) : 'n/a'}`,
      );
    }, 500);
  }, [log, recorder, stopPolling]);

  // -- permissions ----------------------------------------------------------
  useEffect(() => {
    (async () => {
      try {
        const before = await getRecordingPermissionsAsync();
        log(
          `permission before request: granted=${before.granted} status=${before.status} ` +
            `canAskAgain=${before.canAskAgain}`,
        );

        const res = await AudioModule.requestRecordingPermissionsAsync();
        log(
          `permission after request: granted=${res.granted} status=${res.status} ` +
            `canAskAgain=${res.canAskAgain}`,
        );
        setGranted(!!res.granted);

        // Must succeed for the mic to open at all. Previously this sat after
        // the permission call inside the same try, so a permission failure
        // skipped it silently and recording produced an empty file.
        await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
        log('setAudioModeAsync ok (allowsRecording=true, playsInSilentMode=true)');
      } catch (e: any) {
        setGranted(false);
        log(`PERMISSION/AUDIO-MODE FAILED: ${String(e?.message ?? e)}`);
        setError(String(e?.message ?? e));
      }
    })();
    return () => {
      if (tickRef.current) clearInterval(tickRef.current);
      if (pollRef.current) clearInterval(pollRef.current);
      wsRef.current?.close();
      KeepAwake?.deactivateKeepAwake?.();
    };
  }, [log]);

  const startTicking = useCallback(() => {
    startedAtRef.current = Date.now();
    setElapsed(0);
    if (tickRef.current) clearInterval(tickRef.current);
    tickRef.current = setInterval(() => {
      setElapsed((Date.now() - startedAtRef.current) / 1000);
    }, 250);
  }, []);

  const stopTicking = useCallback(() => {
    if (tickRef.current) clearInterval(tickRef.current);
    tickRef.current = null;
  }, []);

  // -- recording ------------------------------------------------------------
  const startRecording = useCallback(async () => {
    if (busyRef.current) return;
    if (phase === 'uploading' || phase === 'processing') return;
    busyRef.current = true;
    try {
      // Gate on the real answer, not just an explicit false. granted === null
      // means the startup request never resolved, and letting that through is
      // exactly how a silent, empty recording gets made.
      let ok = granted === true;
      if (!ok) {
        const res = await AudioModule.requestRecordingPermissionsAsync();
        log(`press-time permission: granted=${res.granted} status=${res.status}`);
        ok = !!res.granted;
        setGranted(ok);
      }
      if (!ok) {
        setError('Microphone permission needed. Enable it in Settings, then record again.');
        setPhase('error');
        busyRef.current = false;
        return;
      }

      setError('');
      setRawNotes('');
      setEnhancedNotes('');
      // Silence the previous take before opening the mic on the next one.
      try {
        rawPlayer.pause();
        enhPlayer.pause();
      } catch {
        /* nothing loaded yet */
      }
      setRawUrl(null);
      setEnhancedUrl(null);
      setServerMs(null);
      setStage('');
      setPct(0);
      setUploadPct(0);

      KeepAwake?.activateKeepAwakeAsync?.();

      // Re-assert the session immediately before every recording. Setting this
      // once at mount is not enough on iOS: the session category can be
      // reconfigured or deactivated underneath us, and a session that is not
      // record-capable at record() time yields a valid .m4a of length zero with
      // no error thrown anywhere -- exactly the symptom being chased.
      try {
        await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
        log('audio mode re-asserted before record');
      } catch (e: any) {
        log(`setAudioModeAsync before record FAILED: ${String(e?.message ?? e)}`);
      }

      // Awaited: record() on an unprepared recorder no-ops silently.
      await recorder.prepareToRecordAsync();
      const prepared: any = recorder.getStatus?.() ?? null;
      log(`after prepare: canRecord=${prepared?.canRecord} url=${prepared?.url}`);

      // Which microphone did iOS actually route to? Inputs can only be queried
      // once prepared. A Bluetooth or headset port here would explain poor
      // room pickup on its own.
      try {
        const inputs: any[] = (recorder.getAvailableInputs?.() as any) ?? [];
        const current: any = await recorder.getCurrentInput?.();
        log(
          `input: "${current?.name}" type=${current?.type} | available: ` +
            (inputs.map((i) => i.name).join(', ') || 'none'),
        );
      } catch (e: any) {
        log(`input query failed: ${String(e?.message ?? e)}`);
      }
      if (prepared && prepared.canRecord === false) {
        setError('The recorder reports it cannot record. Restart the app and try again.');
        setPhase('error');
        busyRef.current = false;
        return;
      }

      recorder.record();
      const started: any = recorder.getStatus?.() ?? null;
      log(
        `after record(): isRecording=${recorder.isRecording} ` +
          `statusIsRecording=${started?.isRecording} canRecord=${started?.canRecord} ` +
          `dur=${started?.durationMillis}ms`,
      );

      setPhase('recording');
      startTicking();
      startPolling();
    } catch (e: any) {
      setError(`Could not start recording: ${String(e?.message ?? e)}`);
      setPhase('error');
    } finally {
      busyRef.current = false;
    }
  }, [granted, log, phase, recorder, startPolling, startTicking]);

  const stopRecording = useCallback(async () => {
    if (phase !== 'recording') return;
    const held = (Date.now() - startedAtRef.current) / 1000;
    stopTicking();
    stopPolling();
    KeepAwake?.deactivateKeepAwake?.();

    // The decisive readings, taken before stop() can muddy them.
    log(
      `hold ended after ${held.toFixed(1)}s: peak durationMillis=${maxDurationRef.current}, ` +
        `metering ever reported=${meterSeenRef.current}`,
    );
    if (maxDurationRef.current <= 0) {
      log('DIAGNOSIS: durationMillis never advanced during the hold -> not capturing');
    }

    let settled: Settled;
    try {
      await recorder.stop();
      settled = await settleRecorder(recorder);
    } catch (e: any) {
      setError(`Could not stop recording: ${String(e?.message ?? e)}`);
      setPhase('error');
      return;
    }

    const { uri, fileSize, mediaServicesDidReset, waitedMs } = settled;
    // Peak duration seen while recording. The post-stop status always reads 0,
    // so it cannot be used to judge whether anything was captured.
    const captured = maxDurationRef.current > 0 ? maxDurationRef.current / 1000 : null;
    log(
      `held ${held.toFixed(1)}s, captured ` +
        `${captured == null ? 'nothing' : captured.toFixed(1) + 's'}, ` +
        `file ${fileSize ?? 'unknown'} bytes, settled in ${waitedMs}ms, ` +
        `reset=${mediaServicesDidReset}`,
    );

    // a stray tap, not a recording
    if (held < 0.7) {
      setPhase('idle');
      setElapsed(0);
      return;
    }

    // Everything below fails here on the phone rather than 100 s later on the
    // server, where the only symptom is ffmpeg's "moov atom not found".
    if (!uri) {
      setError('No audio file was produced. Record again.');
      setPhase('error');
      return;
    }
    if (mediaServicesDidReset) {
      setError(
        'The system interrupted the recording, so the file was never finalised. ' +
          'Keep the app in the foreground and record again.',
      );
      setPhase('error');
      return;
    }
    if (captured == null || captured <= 0) {
      setError(
        `Held for ${held.toFixed(1)}s but the recorder never advanced its ` +
          'duration. The microphone did not capture anything — record again.',
      );
      setPhase('error');
      return;
    }
    if (fileSize != null && fileSize <= 0) {
      setError('The recording file is empty. Record again.');
      setPhase('error');
      return;
    }
    // Size alone would not catch this: an interrupted recording keeps every
    // byte of mdat it managed to write and can still be megabytes.
    if (held >= 5 && captured < held * 0.5) {
      setError(
        `Recording stopped early: ${captured.toFixed(1)}s captured of ` +
          `${held.toFixed(1)}s held. The app was most likely suspended ` +
          'mid-recording. Record again, keeping it in the foreground.',
      );
      setPhase('error');
      return;
    }

    upload(uri);
  }, [log, phase, recorder, stopPolling, stopTicking]);

  // auto-stop safety net
  useEffect(() => {
    if (phase === 'recording' && elapsed >= MAX_RECORD_SECONDS) {
      stopRecording();
    }
  }, [phase, elapsed, stopRecording]);

  // -- upload + live results ------------------------------------------------
  const upload = useCallback(
    async (uri: string) => {
      setPhase('uploading');
      setStage('Connecting');
      setPct(0);

      let b64 = '';
      try {
        b64 = await FS.readAsStringAsync(uri, { encoding: 'base64' });
      } catch (e: any) {
        setError(`Could not read the recording: ${String(e?.message ?? e)}`);
        setPhase('error');
        return;
      }

      let ws: WebSocket;
      try {
        ws = new WebSocket(SERVER_URL);
      } catch (e: any) {
        setError(`Could not open ${SERVER_URL}: ${String(e?.message ?? e)}`);
        setPhase('error');
        return;
      }
      wsRef.current = ws;

      ws.onopen = async () => {
        try {
          ws.send(JSON.stringify({ type: 'start', ext: EXT }));
          for (let i = 0; i < b64.length; i += CHUNK_CHARS) {
            if (wsRef.current !== ws) return; // torn down underneath us
            ws.send(b64.slice(i, i + CHUNK_CHARS));
            setUploadPct(Math.min(100, Math.round(((i + CHUNK_CHARS) / b64.length) * 100)));
            // let the UI breathe between frames
            await new Promise((r) => setTimeout(r, 0));
          }
          ws.send(JSON.stringify({ type: 'end' }));
          setUploadPct(100);
          setPhase('processing');
          setStage('Uploaded, waiting for the server');
          startTicking();
        } catch (e: any) {
          setError(`Upload failed: ${String(e?.message ?? e)}`);
          setPhase('error');
          stopTicking();
        }
      };

      ws.onmessage = (ev) => {
        let msg: any;
        try {
          msg = JSON.parse(String(ev.data));
        } catch {
          return;
        }

        if (msg.type === 'error') {
          setError(String(msg.message ?? 'The server reported an error.'));
          phaseRef.current = 'error';
          setPhase('error');
          stopTicking();
          ws.close();
          return;
        }

        if (msg.type === 'status') {
          if (typeof msg.stage === 'string') setStage(stageLabel(msg.stage));
          if (typeof msg.pct === 'number') setPct(msg.pct);
        }

        // Render whatever has landed, on whichever frame it lands. The raw path
        // finishes first, so raw_notes may arrive before enhanced_notes.
        if (typeof msg.raw_notes === 'string' && msg.raw_notes) setRawNotes(msg.raw_notes);
        if (typeof msg.enhanced_notes === 'string' && msg.enhanced_notes) {
          setEnhancedNotes(msg.enhanced_notes);
        }
        if (typeof msg.raw_audio_url === 'string') setRawUrl(msg.raw_audio_url);
        if (typeof msg.enhanced_audio_url === 'string') setEnhancedUrl(msg.enhanced_audio_url);
        if (typeof msg.ms === 'number') setServerMs(msg.ms);

        if (msg.type === 'result') {
          setPct(100);
          setStage('Done');
          phaseRef.current = 'done';
          setPhase('done');
          stopTicking();
          ws.close();
        }
      };

      ws.onerror = () => {
        if (wsRef.current !== ws) return;
        if (phaseRef.current !== 'uploading' && phaseRef.current !== 'processing') return;
        setError(`Lost the connection to ${SERVER_URL}. Is the server running?`);
        phaseRef.current = 'error';
        setPhase('error');
        stopTicking();
      };

      ws.onclose = () => {
        if (wsRef.current !== ws) return;
        if (phaseRef.current === 'uploading' || phaseRef.current === 'processing') {
          setError('The server closed the connection before sending results.');
          setPhase('error');
          stopTicking();
        }
      };
    },
    [startTicking, stopTicking],
  );

  // -- derived --------------------------------------------------------------
  const recording = phase === 'recording';
  const working = phase === 'uploading' || phase === 'processing';
  const barPct = phase === 'uploading' ? uploadPct : pct;

  let indicator = 'Ready';
  if (recording) indicator = 'Recording';
  else if (phase === 'uploading') indicator = 'Sending';
  else if (phase === 'processing') indicator = 'Working';
  else if (phase === 'done') indicator = 'Done';
  else if (phase === 'error') indicator = 'Error';
  else if (granted === false) indicator = 'No mic';

  let statusLine = 'Hold the button and speak';
  if (recording) statusLine = 'Recording';
  else if (phase === 'uploading') statusLine = 'Sending audio';
  else if (phase === 'processing') statusLine = stage || 'Working';
  else if (phase === 'done') {
    statusLine = serverMs != null ? `Done in ${(serverMs / 1000).toFixed(1)}s` : 'Done';
  } else if (phase === 'error') statusLine = 'Something went wrong';

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: PAGE }}>
      <StatusBar barStyle="dark-content" backgroundColor={PAGE} />
      <View
        style={{
          flex: 1,
          paddingHorizontal: 18,
          paddingTop: Platform.OS === 'android' ? 26 : 4,
        }}
      >
        {/* ---------------- header ---------------- */}
        <View
          style={{
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'space-between',
            paddingBottom: 16,
          }}
        >
          <View style={{ flexDirection: 'row', alignItems: 'center' }}>
            <View
              style={{
                width: 13,
                height: 13,
                borderRadius: 6.5,
                backgroundColor: INK,
                flexDirection: 'row',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <View
                style={{
                  width: 3.4,
                  height: 3.4,
                  borderRadius: 1.7,
                  backgroundColor: EMBER,
                  marginRight: 1.6,
                }}
              />
              <View
                style={{ width: 3.4, height: 3.4, borderRadius: 1.7, backgroundColor: EMBER }}
              />
            </View>
            <Text style={{ marginLeft: 9, fontSize: 15, fontWeight: '600', color: INK }}>
              Meeting notes
            </Text>
          </View>

          {/* recording indicator */}
          <View
            style={{
              flexDirection: 'row',
              alignItems: 'center',
              borderWidth: 3,
              borderColor: INK,
              borderRadius: 11,
              paddingHorizontal: 9,
              paddingVertical: 3,
              backgroundColor: PAPER,
            }}
          >
            <View
              style={{
                width: 7,
                height: 7,
                borderRadius: 3.5,
                marginRight: 6,
                backgroundColor: recording ? RED : working ? INK : MUTED,
                borderWidth: recording || working ? 0 : 1.5,
                borderColor: LABEL_GREY,
              }}
            />
            <Text
              style={{
                fontSize: 11,
                fontWeight: '700',
                letterSpacing: 0.8,
                color: INK,
                textTransform: 'uppercase',
              }}
            >
              {indicator}
            </Text>
          </View>
        </View>

        {/* ---------------- status band ---------------- */}
        <View style={{ paddingBottom: 18 }}>
          <Text
            style={{
              fontSize: 11,
              fontWeight: '700',
              letterSpacing: 0.9,
              color: LABEL_GREY,
              textTransform: 'uppercase',
            }}
            numberOfLines={1}
          >
            {statusLine}
          </Text>

          <View
            style={{
              flexDirection: 'row',
              alignItems: 'flex-end',
              justifyContent: 'space-between',
              marginTop: 2,
            }}
          >
            <Text style={{ fontSize: 30, fontWeight: '500', color: INK, letterSpacing: -0.5 }}>
              {clock(elapsed)}
            </Text>
            <Text style={{ fontSize: 13, fontWeight: '600', color: INK, paddingBottom: 4 }}>
              {barPct}%
            </Text>
          </View>

          <View
            style={{
              height: 14,
              marginTop: 8,
              backgroundColor: MUTED,
              borderWidth: 3,
              borderColor: INK,
              borderRadius: 7,
              justifyContent: 'center',
              overflow: 'hidden',
            }}
          >
            <View
              style={{
                height: 8,
                marginHorizontal: 3,
                width: `${Math.max(0, Math.min(100, barPct))}%`,
                backgroundColor: INK,
                borderRadius: 4,
              }}
            />
          </View>

          {!!error && (
            <Text style={{ marginTop: 10, fontSize: 13, fontWeight: '600', color: INK }}>
              {error}
            </Text>
          )}

          {/* live recorder diagnostics — also mirrored to the Metro console */}
          {!!diag && (
            <Text
              style={{ marginTop: 8, fontSize: 11, color: LABEL_GREY }}
              numberOfLines={2}
            >
              {diag}
            </Text>
          )}
        </View>

        {/* ---------------- RAW ---------------- */}
        <Pane
          label="RAW"
          accent={false}
          body={rawNotes}
          placeholder={working ? 'The raw pass lands first' : 'Record something to see the raw notes'}
          audioUrl={rawUrl}
          playing={rawPlayback.playing}
          onTogglePlay={() => togglePlay('raw')}
        />

        <View style={{ height: 20 }} />

        {/* ---------------- ENHANCED ---------------- */}
        <Pane
          label="ENHANCED"
          accent
          body={enhancedNotes}
          placeholder={working ? 'Still cleaning the audio' : 'The enhanced notes appear here'}
          audioUrl={enhancedUrl}
          playing={enhPlayback.playing}
          onTogglePlay={() => togglePlay('enh')}
        />

        {/* ---------------- button ---------------- */}
        <View style={{ marginTop: 26, marginBottom: Platform.OS === 'android' ? 20 : 8 }}>
          {/* the entire depth effect: a solid ink rect offset 8px below */}
          <View
            style={{
              position: 'absolute',
              left: 0,
              right: 0,
              top: 8,
              height: 42,
              borderRadius: 21,
              backgroundColor: INK,
            }}
          />
          <Pressable
            onPressIn={startRecording}
            onPressOut={stopRecording}
            disabled={working}
            style={{
              height: 42,
              borderRadius: 21,
              borderWidth: 4,
              borderColor: INK,
              backgroundColor: recording ? RED : working ? MUTED : GREEN,
              flexDirection: 'row',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {/* mic glyph */}
            <View
              style={{
                width: 17,
                height: 17,
                borderRadius: 8.5,
                backgroundColor: PAPER,
                borderWidth: 2.5,
                borderColor: INK,
                alignItems: 'center',
                justifyContent: 'center',
                marginRight: 9,
              }}
            >
              <View style={{ width: 4, height: 7, borderRadius: 2, backgroundColor: INK }} />
            </View>
            <Text
              style={{
                fontSize: 15,
                fontWeight: '500',
                letterSpacing: 1,
                color: working ? INK : PAPER,
              }}
            >
              {recording ? 'RELEASE TO SEND' : working ? 'WORKING' : 'HOLD TO RECORD'}
            </Text>
          </Pressable>
        </View>
      </View>
    </SafeAreaView>
  );
}

// ---------------------------------------------------------------------------
const BAR_WIDTHS = ['92%', '76%', '85%', '61%'];

/**
 * Play/stop pill for one pane's clip. Purely presentational -- both players
 * live in App so that starting one can stop the other. Two panes each owning
 * a private player would let the raw and enhanced clips talk over each other,
 * which is exactly what an A/B must not do.
 */
function PlayButton({
  url,
  accent,
  playing,
  onPress,
}: {
  url: string | null;
  accent: boolean;
  playing: boolean;
  onPress: () => void;
}) {
  if (!url) return null;

  const fill = accent ? EMBER : INK;
  return (
    <Pressable
      onPress={onPress}
      hitSlop={12}
      style={{
        width: 26,
        height: 26,
        borderRadius: 13,
        backgroundColor: fill,
        borderWidth: accent ? 3 : 0,
        borderColor: INK,
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {playing ? (
        <View style={{ flexDirection: 'row' }}>
          <View
            style={{
              width: 3,
              height: 10,
              borderRadius: 1,
              backgroundColor: PAPER,
              marginRight: 3,
            }}
          />
          <View style={{ width: 3, height: 10, borderRadius: 1, backgroundColor: PAPER }} />
        </View>
      ) : (
        // CSS-style triangle, so no SVG dependency. Nudged right so its optical
        // centre sits in the middle of the circle.
        <View
          style={{
            width: 0,
            height: 0,
            borderTopWidth: 5,
            borderBottomWidth: 5,
            borderLeftWidth: 8,
            borderTopColor: 'transparent',
            borderBottomColor: 'transparent',
            borderLeftColor: PAPER,
            marginLeft: 3,
          }}
        />
      )}
    </Pressable>
  );
}

function Pane({
  label,
  accent,
  body,
  placeholder,
  audioUrl,
  playing,
  onTogglePlay,
}: {
  label: string;
  accent: boolean;
  body: string;
  placeholder: string;
  audioUrl: string | null;
  playing: boolean;
  onTogglePlay: () => void;
}) {
  return (
    // Outer view stays unclipped so the badge can overlap the corner; the inner
    // view does the clipping so long notes cannot escape the rounded border.
    <View>
      <View
        style={{
          height: PANE_HEIGHT,
          backgroundColor: accent ? ENHANCED_FILL : PAPER,
          borderRadius: 16,
          borderWidth: accent ? 4 : 3.5,
          borderColor: accent ? EMBER : INK,
          overflow: 'hidden',
        }}
      >
        <ScrollView
          style={{ flex: 1 }}
          contentContainerStyle={{
            paddingTop: 20,
            paddingHorizontal: 14,
            paddingBottom: 16,
          }}
          showsVerticalScrollIndicator
        >
          {body ? (
            <Text style={{ fontSize: 13, lineHeight: 20, color: accent ? INK : GREY_TEXT }}>
              {body}
            </Text>
          ) : (
            <View>
              <Text
                style={{
                  fontSize: 11,
                  fontWeight: '700',
                  letterSpacing: 0.9,
                  color: LABEL_GREY,
                  textTransform: 'uppercase',
                  marginBottom: 12,
                }}
              >
                {placeholder}
              </Text>
              {BAR_WIDTHS.map((w) => (
                <View
                  key={w}
                  style={{
                    height: 12,
                    width: w as any,
                    borderRadius: 6,
                    backgroundColor: MUTED,
                    marginBottom: 8,
                  }}
                />
              ))}
            </View>
          )}
        </ScrollView>
      </View>

      {/* badge, overlapping the top-left corner */}
      <View
        style={{
          position: 'absolute',
          top: -11,
          left: 14,
          backgroundColor: accent ? EMBER : INK,
          borderRadius: 11,
          paddingHorizontal: 10,
          paddingVertical: 3,
          borderWidth: accent ? 3 : 0,
          borderColor: INK,
        }}
      >
        <Text
          style={{
            fontSize: 11,
            fontWeight: '700',
            letterSpacing: 1,
            color: PAPER,
          }}
        >
          {label}
        </Text>
      </View>

      {/* play button, mirroring the badge on the opposite corner */}
      <View style={{ position: 'absolute', top: -13, right: 14 }}>
        <PlayButton
          url={audioUrl}
          accent={accent}
          playing={playing}
          onPress={onTogglePlay}
        />
      </View>
    </View>
  );
}
