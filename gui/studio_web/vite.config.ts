import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// `npm run dev` proxies the API to a running `opencam-studio --no-browser` (port 8765).
// `npm run build` writes the bundle the Python package serves (committed, so users need no Node).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: '../opencam_gui/studio/static',
    emptyOutDir: true,
  },
  server: {
    proxy: { '/api': 'http://127.0.0.1:8765' },
  },
})
