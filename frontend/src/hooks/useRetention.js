import { useEffect, useRef, useState } from 'react';
import { useAuth } from '../components/Providers';

export function useRetention() {
  const { retainImages } = useAuth();
  const [retain, setValue] = useState(retainImages);
  const changed = useRef(false);
  useEffect(() => {
    if (!changed.current) setValue(retainImages);
  }, [retainImages]);
  function setRetain(value) {
    changed.current = true;
    setValue(value);
  }
  return [retain, setRetain];
}
