/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      // Restrained and technical. An instrument panel, not a landing page: no
      // gradients, no glass, no rounded pastel cards. Exactly two accents — one for
      // the winning path, one for rejection — so colour always means something.
      colors: {
        ink: "#08090b",
        panel: "#0e1116",
        raised: "#141922",
        line: "#1f2630",
        dim: "#5c6773",
        body: "#aeb8c4",
        bright: "#e6ebf1",
        win: "#5eead4",
        reject: "#fb7185",
        caution: "#fbbf24",
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
    },
  },
  plugins: [],
};
