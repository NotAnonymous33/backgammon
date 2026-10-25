// Override with VITE_BACKEND_URL (e.g. in .env.local). By default the backend is assumed
// to run on port 5000 of whichever host served this page, which also works for `npm run host`.
export const BACKEND_URL: string =
    import.meta.env.VITE_BACKEND_URL ?? `${window.location.protocol}//${window.location.hostname}:5000`
