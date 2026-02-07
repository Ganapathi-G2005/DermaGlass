/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                brand: {
                    primary: '#0284c7', // Sky Blue
                    secondary: '#f1f5f9', // Slate 100
                    accent: '#fef08a', // Yellow 200 (Warning)
                    dark: '#0f172a', // Slate 900
                }
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
            }
        },
    },
    plugins: [
        require('@tailwindcss/typography'),
    ],
}
