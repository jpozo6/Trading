/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    darkMode: 'class', // start with class, though we are forcing dark mode in body usually
    theme: {
        extend: {
            colors: {
                gray: {
                    950: '#030712',
                    900: '#111827',
                    850: '#1f2937',
                    800: '#374151',
                    700: '#4b5563',
                    // ... standard palette
                },
                brand: {
                    500: '#3b82f6', // bright blue
                    600: '#2563eb',
                    400: '#60a5fa',
                },
                accent: {
                    500: '#10b981', // emerald
                }
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
            }
        },
    },
    plugins: [
        require('@tailwindcss/forms'),
    ],
}
