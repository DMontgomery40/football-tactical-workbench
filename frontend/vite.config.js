import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const frontendDir = fileURLToPath(new URL('.', import.meta.url));
const repoRoot = path.resolve(frontendDir, '..');
const contractsDir = path.resolve(repoRoot, 'packages/contracts');

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@contracts': contractsDir,
    },
  },
  server: {
    host: '0.0.0.0',
    port: 4317,
    strictPort: true,
    fs: {
      allow: [repoRoot],
    },
  },
});
