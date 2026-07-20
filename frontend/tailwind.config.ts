import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#FAF6F1",      // warm off-white
        surface: "#FFFFFF",
        "surface-alt": "#F0E6DA",  // light warm beige for cards/sections
        brown: {
          900: "#3B2A1E",          // primary text / headlines
          700: "#6B4A32",          // primary brand / buttons
          500: "#9C6B45",          // accents, active states
          300: "#C9A87C",          // borders, secondary accents
        },
        success: "#4C7A54",
        warning: "#B9772F",
        danger: "#A2402D",
      },
      fontFamily: {
        sans: ["Outfit", "Inter", "sans-serif"],
      },
    },
  },
  plugins: [],
};
export default config;
