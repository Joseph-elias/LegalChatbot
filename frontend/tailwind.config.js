/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f6ff',
          100: '#e0ecff',
          200: '#c7dcff',
          300: '#9ec5ff',
          400: '#6ba4ff',
          500: '#3b82ff',
          600: '#1a65f7',
          700: '#1a52e9',
          800: '#1a43bc',
          900: '#1a365d',
          950: '#0f1d38',
        },
        secondary: {
          50: '#faf8eb',
          100: '#f5efd0',
          200: '#eadda3',
          300: '#e0c76c',
          400: '#d4af37', // Gold
          500: '#c29b29',
          600: '#a17b21',
          700: '#7e591e',
          800: '#6c4a1f',
          900: '#5d3f1f',
          950: '#362111',
        },
        accent: {
          50: '#fcf5f9',
          100: '#f9eaf3',
          200: '#f5d5e7',
          300: '#edb3d3',
          400: '#e186b8',
          500: '#d45f9e',
          600: '#c13f84',
          700: '#ab2a6e',
          800: '#9f2b68',
          900: '#71224c',
          950: '#450c2a',
        },
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out forwards',
        'typing': 'pulse 1.5s infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: 0, transform: 'translateY(10px)' },
          '100%': { opacity: 1, transform: 'translateY(0)' },
        },
        pulse: {
          '0%, 100%': { opacity: 0.6, transform: 'scale(1)' },
          '50%': { opacity: 1, transform: 'scale(1.05)' },
        },
      },
      fontFamily: {
        sans: ['Tajawal', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
      },
    },
  },
  plugins: [],
};