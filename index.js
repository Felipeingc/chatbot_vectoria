require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const path = require('path');

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(cors());
// Aumentamos el límite del payload JSON para acomodar historiales de conversación más largos
app.use(express.json({ limit: '5mb' })); // <--- CAMBIO: Límite aumentado
app.use(express.static(__dirname)); // Sirve archivos estáticos desde el directorio actual

// Configuración de Gemini AI
if (!process.env.GEMINI_API_KEY) {
  console.error("Error crítico: La variable de entorno GEMINI_API_KEY no está definida.");
  process.exit(1); // Termina el proceso si la API key no está
}
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Endpoint para el streaming de respuestas del chatbot
app.post('/stream', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders(); // Envía las cabeceras inmediatamente

  try {
    // --- INICIO DE CAMBIOS PARA PERSISTENCIA ---
    // 1. Recibir el historial completo del cliente.
    //    El frontend ahora envía un objeto { history: [...] }
    //    donde `history` es un array de objetos { role: "user"|"model", parts: [{text: "..."}] }
    const clientConversationHistory = req.body.history || [];

    // 2. Validación básica del historial recibido
    if (!clientConversationHistory.length) {
      // Si no hay historial (no debería pasar si el frontend siempre envía algo)
      if (!res.writableEnded) {
        res.write(`data: ${JSON.stringify({ error: "No se recibió historial de conversación." })}\n\n`);
        res.write('data: [END]\n\n');
        res.end();
      }
      return;
    }

    const lastMessage = clientConversationHistory[clientConversationHistory.length - 1];
    if (!lastMessage || !lastMessage.parts || !lastMessage.parts[0] || !lastMessage.parts[0].text || !lastMessage.parts[0].text.trim()) {
      // Si el último mensaje (la pregunta actual del usuario) está vacío
      if (!res.writableEnded) {
        res.write(`data: ${JSON.stringify({ texto: "La pregunta no puede estar vacía." })}\n\n`);
        res.write('data: [END]\n\n');
        res.end();
      }
      return;
    }

    // 3. El historial del cliente ya está en el formato que Gemini necesita para `contents`
    const contentsForGemini = clientConversationHistory;
    // --- FIN DE CAMBIOS PARA PERSISTENCIA ---

    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash-preview-04-17" }); // O el modelo específico que prefieras

    // 4. Pasar el historial completo (`contentsForGemini`) al modelo
    const streamingResp = await model.generateContentStream({ contents: contentsForGemini });

    let streamClosed = false; // Flag para controlar el cierre del stream

    // Asegurar que el stream se cierre si el cliente se desconecta
    req.on('close', () => {
        // console.log('Cliente desconectado, cancelando stream si es posible.');
        // Aquí podrías intentar cancelar el stream de Gemini si el SDK lo permite y es necesario,
        // por ahora, solo marcamos que se cerró para no intentar escribir más.
        streamClosed = true;
    });

    for await (const chunk of streamingResp.stream) {
      if (streamClosed || res.writableEnded) break; // Salir si el cliente se desconectó o el stream ya terminó

      let texto = '';
      if (chunk && typeof chunk.text === 'function') {
        texto = chunk.text();
      } else if (!texto && chunk && chunk.candidates?.length > 0 &&
               chunk.candidates[0].content?.parts?.length > 0 &&
               typeof chunk.candidates[0].content.parts[0].text === 'string') {
        texto = chunk.candidates[0].content.parts[0].text;
      }

      if (typeof texto === "string" && texto.trim() !== "") {
        if (!streamClosed && !res.writableEnded) {
            res.write(`data: ${JSON.stringify({ texto })}\n\n`);
        }
      }
    }

    // Señal de finalización del stream
    if (!streamClosed && !res.writableEnded) {
        res.write('data: [END]\n\n');
        res.end();
    }

  } catch (error) {
    console.error("Error en el endpoint /stream:", error);
    if (!res.headersSent) {
        // Esto es improbable debido a flushHeaders(), pero es una buena práctica verificar
        res.setHeader('Content-Type', 'text/event-stream'); // Reasegurar si no se enviaron
    }
    if (!res.writableEnded) { // Solo escribir si el stream no ha terminado
        try {
            res.write(`data: ${JSON.stringify({ error: 'Error del servidor al procesar la solicitud: ' + error.message })}\n\n`);
            res.write('data: [END]\n\n');
        } catch (e) {
            console.error("Error al escribir el mensaje de error en el stream:", e);
        } finally {
            if (!res.writableEnded) {
                res.end();
            }
        }
    }
  }
});

// Ruta para servir el archivo HTML principal
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Inicia el servidor
app.listen(port, () => {
  console.log(`Servidor escuchando en http://localhost:${port}`);
  console.log('Asegúrese de que GEMINI_API_KEY está configurada en su archivo .env');
});