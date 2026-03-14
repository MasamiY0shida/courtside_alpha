/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow the dashboard to proxy API calls to the Rust backend (port 4000)
  async rewrites() {
    return [
      // Rust execution engine (port 4000)
      {
        source:      '/api/trades/:path*',
        destination: 'http://localhost:4000/trades/:path*',
      },
      {
        source:      '/api/wallet',
        destination: 'http://localhost:4000/wallet',
      },
      // Python server.py (port 8000)
      {
        source:      '/api/games/:path*',
        destination: 'http://localhost:8000/games/:path*',
      },
      {
        source:      '/api/games',
        destination: 'http://localhost:8000/games',
      },
      {
        source:      '/api/markets',
        destination: 'http://localhost:8000/markets',
      },
      {
        source:      '/api/signals/:path*',
        destination: 'http://localhost:8000/signals/:path*',
      },
      {
        source:      '/api/signals',
        destination: 'http://localhost:8000/signals',
      },
    ];
  },
};

module.exports = nextConfig;
