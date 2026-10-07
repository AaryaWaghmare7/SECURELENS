import { useEffect, useRef, useState } from 'react';
import { api, analysisForm } from '../services/api';

export function useCamera() {
  const video = useRef(null);
  const stream = useRef(null);
  const active = useRef(false);
  const controller = useRef(null);
  const timer = useRef(null);
  const mounted = useRef(true);
  const generation = useRef(0);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [count, setCount] = useState(0);
  const lastFrame = useRef(null);
  function stop() {
    active.current = false;
    generation.current += 1;
    clearTimeout(timer.current);
    controller.current?.abort();
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
    if (mounted.current) setRunning(false);
  }
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      stop();
    };
  }, []);
  async function capture() {
    const feed = video.current;
    if (!feed?.videoWidth) throw new Error('Camera is not ready yet.');
    const canvas = document.createElement('canvas');
    canvas.width = Math.min(feed.videoWidth, 960);
    canvas.height = Math.round((feed.videoHeight * canvas.width) / feed.videoWidth);
    canvas.getContext('2d').drawImage(feed, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
    if (!blob) throw new Error('Could not capture this frame.');
    return new File([blob], `camera-${Date.now()}.png`, { type: 'image/png' });
  }
  async function tick(run) {
    if (!active.current || run !== generation.current) return;
    try {
      const file = await capture();
      if (!active.current || !mounted.current || run !== generation.current) return;
      controller.current = new AbortController();
      const next = await api.analyze(
        analysisForm([file], { save: false, source: 'webcam' }),
        controller.current.signal,
      );
      if (active.current && mounted.current && run === generation.current) {
        lastFrame.current = file;
        setResult(next);
        setCount((value) => value + 1);
      }
    } catch (e) {
      if (e.name !== 'AbortError' && active.current && run === generation.current) {
        setError(e.message);
        stop();
        return;
      }
    }
    if (active.current && run === generation.current)
      timer.current = setTimeout(() => tick(run), 3000);
  }
  async function start() {
    if (active.current) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera access needs localhost or HTTPS and a supported browser.');
      return;
    }
    const run = ++generation.current;
    active.current = true;
    lastFrame.current = null;
    setResult(null);
    setCount(0);
    setError('');
    setRunning(true);
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' },
        audio: false,
      });
      if (!active.current || !mounted.current || run !== generation.current) {
        media.getTracks().forEach((track) => track.stop());
        return;
      }
      stream.current = media;
      video.current.srcObject = media;
      await video.current.play();
      await tick(run);
    } catch {
      if (active.current && run === generation.current) {
        setError('Camera access was denied or is unavailable. Allow access, or use image upload.');
        stop();
      }
    }
  }
  return { video, running, result, error, count, start, stop, capture, lastFrame };
}
