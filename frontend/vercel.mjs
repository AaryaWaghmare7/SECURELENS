import { createVercelConfig } from './deployment/config.mjs';

export const config = createVercelConfig(process.env);
