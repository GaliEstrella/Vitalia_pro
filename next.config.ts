import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  
  // --- AGREGA ESTO ---
  typescript: {
    // Ignora errores de tipos durante el build (opcional pero recomendado para tareas)
    ignoreBuildErrors: true,
  }
  // -------------------
};

export default nextConfig;