# `api/`

This directory is reserved for [Vercel Serverless / Edge Functions](https://vercel.com/docs/functions).

Vercel auto-discovers any file under `api/` at deploy time and exposes it as an
HTTP endpoint at the matching `/api/...` route (e.g. `api/submit-evaluation.ts`
is served at `/api/submit-evaluation`). The location is a Vercel convention —
files outside `api/` are not deployed as functions.

Do not put general-purpose scripts or library code here; those belong in
`src/scripts/` or `src/` respectively.
