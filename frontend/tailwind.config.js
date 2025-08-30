/** @type {import('tailwindcss').Config} */
import typography from '@tailwindcss/typography';
import forms from '@tailwindcss/forms';

export default {
  content: ['./src/**/*.{html,js,svelte,ts}'],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          'Inter',
          'system-ui',
          '-apple-system',
          'Segoe UI',
          'sans-serif'
        ],
        mono: [
          'JetBrains Mono',
          'SF Mono',
          'Consolas',
          'monospace'
        ],
        display: [
          'Inter',
          'system-ui',
          'sans-serif'
        ]
      },
      colors: {
        // Light mode colors
        light: {
          bg: {
            primary: 'oklch(100% 0 0)',     // Pure white
            secondary: 'oklch(98% 0 0)',    // Off-white
            tertiary: 'oklch(96% 0 0)',     // Light gray
          },
          text: {
            primary: 'oklch(15% 0 0)',      // Near black
            secondary: 'oklch(45% 0 0)',    // Medium gray
            tertiary: 'oklch(65% 0 0)',     // Light gray
          },
          border: {
            primary: 'oklch(85% 0 0)',      // Light border
            secondary: 'oklch(90% 0 0)',    // Very light border
          }
        },
        
        // Dark mode colors
        dark: {
          bg: {
            primary: 'oklch(11% 0 0)',      // #1A1A1A (your current bg)
            secondary: 'oklch(8% 0 0)',     // #0D0D0D (your current bg-dark)
            tertiary: 'oklch(16% 0 0)',     // #262626 (your current bg-light)
          },
          text: {
            primary: 'oklch(95% 0 0)',      // Near white (your current text)
            secondary: 'oklch(70% 0 0)',    // Medium gray (your current text-muted)
            tertiary: 'oklch(55% 0 0)',     // Darker gray
          },
          border: {
            primary: 'oklch(25% 0 0)',      // Dark border
            secondary: 'oklch(20% 0 0)',    // Darker border
          }
        },

        // Semantic colors (work in both themes)
        success: {
          DEFAULT: 'oklch(64% 0.18 145)',  // Green
          light: 'oklch(94% 0.05 145)',    // Light green bg
          dark: 'oklch(25% 0.08 145)',     // Dark green bg
        },
        warning: {
          DEFAULT: 'oklch(68% 0.19 85)',   // Yellow
          light: 'oklch(96% 0.03 85)',     // Light yellow bg
          dark: 'oklch(25% 0.06 85)',      // Dark yellow bg
        },
        error: {
          DEFAULT: 'oklch(63% 0.18 25)',   // Red
          light: 'oklch(96% 0.03 25)',     // Light red bg
          dark: 'oklch(25% 0.06 25)',      // Dark red bg
        },
        info: {
          DEFAULT: 'oklch(64% 0.18 250)',  // Blue
          light: 'oklch(96% 0.03 250)',    // Light blue bg
          dark: 'oklch(25% 0.06 250)',     // Dark blue bg
        },
        
        // Brand colors (theme-aware)
        primary: {
          PLACEHOLDER: 'oklch(100% 0 0);'
        },
        secondary: {
          PLACEHOLDER: 'oklch(100% 0 0);'
        }
      },
      animation: {
        // Fade animations
        'fade-in': 'fadeIn 0.3s ease-out',
        'fade-in-slow': 'fadeIn 0.6s ease-out',
        'fade-out': 'fadeOut 0.2s ease-in',
        
        // Slide animations
        'slide-up': 'slideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
        'slide-down': 'slideDown 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
        'slide-left': 'slideLeft 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
        'slide-right': 'slideRight 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
        
        // Scale animations
        'scale-in': 'scaleIn 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
        'scale-out': 'scaleOut 0.15s cubic-bezier(0.4, 0, 1, 1)',
        
        // Pulse variations
        'pulse-slow': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pulse-fast': 'pulse 1s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        
        // Loading animations
        'spin-slow': 'spin 2s linear infinite',
        'bounce-subtle': 'bounceSubtle 1s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' }
        },
        fadeOut: {
          '0%': { opacity: '1' },
          '100%': { opacity: '0' }
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' }
        },
        slideDown: {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' }
        },
        slideLeft: {
          '0%': { transform: 'translateX(10px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' }
        },
        slideRight: {
          '0%': { transform: 'translateX(-10px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' }
        },
        scaleIn: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' }
        },
        scaleOut: {
          '0%': { transform: 'scale(1)', opacity: '1' },
          '100%': { transform: 'scale(0.95)', opacity: '0' }
        },
        bounceSubtle: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-2px)' }
        }
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '112': '28rem',
        '128': '32rem',
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
        '4xl': '2rem',
      },
      boxShadow: {
        'soft': '0 2px 15px -3px rgba(0, 0, 0, 0.07), 0 10px 20px -2px rgba(0, 0, 0, 0.04)',
        'medium': '0 4px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 30px -5px rgba(0, 0, 0, 0.05)',
        'hard': '0 10px 40px -10px rgba(0, 0, 0, 0.15), 0 20px 50px -10px rgba(0, 0, 0, 0.1)',
      }
    },
  },
  plugins: [
    typography,
    forms
  ],
}
