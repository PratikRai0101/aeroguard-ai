# AeroGuard AI Mobile App

Cross-platform React Native mobile app built with Expo. It connects to the AeroGuard AI FastAPI backend and includes a dashboard + AI chatbot.

## Features

- **Dashboard**: Real-time AQI, temperature, humidity, gas, trend, outdoor AQI, preventive measures, and recent readings.
- **AI Chat**: Ask questions about the air quality data. The chatbot is powered by a local Qwen 3.5 2B model via Ollama.

## Quick start

1. Start the backend in one terminal (`python setup_and_run.py` from the project root) and keep it running.
2. Open a **second terminal** and set the backend API URL:

   ```bash
   # On the same machine (simulator / Expo Go)
   echo "EXPO_PUBLIC_API_URL=http://localhost:8000" > mobile/.env

   # On a real device connected to the same WiFi
   echo "EXPO_PUBLIC_API_URL=http://192.168.1.XXX:8000" > mobile/.env
   ```

3. Install dependencies and start:

   ```bash
   cd mobile
   npm install
   npx expo start
   ```

4. Scan the QR code with the Expo Go app (iOS/Android) or run on a simulator.

## Project structure

```
mobile/
├── app/
│   ├── (tabs)/
│   │   ├── _layout.tsx      # Bottom tab navigation
│   │   ├── index.tsx        # Dashboard screen
│   │   └── chat.tsx         # AI chat screen
│   └── _layout.tsx          # Root layout
├── lib/
│   └── api.ts               # Backend API client
├── .env.example             # Example environment variables
└── app.json                 # Expo configuration
```

## Configuration

Create `mobile/.env` from `.env.example` and set `EXPO_PUBLIC_API_URL` to the backend address.

| Variable | Description |
|----------|-------------|
| `EXPO_PUBLIC_API_URL` | URL of the FastAPI backend (e.g., `http://192.168.1.10:8000`) |

## Build for production

Use EAS (Expo Application Services) to build for iOS/Android:

```bash
npx eas build --platform ios    # or android
```

For local builds, see the [Expo local build docs](https://docs.expo.dev/build-reference/local-builds/).
