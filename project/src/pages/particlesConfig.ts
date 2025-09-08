import type { ISourceOptions } from "tsparticles-engine";

export const options: ISourceOptions = {
  background: {
    color: {
      value: "#f0f7ff", // Un azul muy claro de fondo
    },
  },
  fpsLimit: 120,
  interactivity: {
    events: {
      onHover: {
        enable: true,
        mode: "repulse", // Las partículas se alejan del cursor
      },
      resize: true,
    },
    modes: {
      repulse: {
        distance: 80,
        duration: 0.4,
      },
    },
  },
  particles: {
    color: {
      value: "#2563eb", // Color azul principal para las partículas
    },
    links: {
      color: "#3b82f6", // Color azul para las líneas que conectan las partículas
      distance: 150,
      enable: true,
      opacity: 0.3,
      width: 1,
    },
    move: {
      direction: "none",
      enable: true,
      outModes: {
        default: "bounce",
      },
      random: false,
      speed: 1,
      straight: false,
    },
    number: {
      density: {
        enable: true,
        area: 800,
      },
      value: 80, // Número de partículas en la pantalla
    },
    opacity: {
      value: 0.3,
    },
    shape: {
      type: "circle",
    },
    size: {
      value: { min: 1, max: 5 },
    },
  },
  detectRetina: true,
};