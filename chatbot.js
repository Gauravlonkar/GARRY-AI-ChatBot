import dotenv from "dotenv";
import express from "express";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {
    ChatPromptTemplate,
    MessagesPlaceholder,
} from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// ✅ Model
const model = new ChatGoogleGenerativeAI({
    model: "gemini-2.5-flash-lite",
    temperature: 0,
    apiKey: process.env.GOOGLE_API_KEY,
});

// ✅ Prompt
const prompt = ChatPromptTemplate.fromMessages([
    [
        "system",
        "You are a helpful and friendly AI assistant. Be concise and clear. Format responses with markdown when helpful.",
    ],
    new MessagesPlaceholder("history"),
    ["human", "{input}"],
]);

// ✅ Chain
const chain = RunnableSequence.from([prompt, model]);

// ✅ Per-session chat history
const sessions = {};

function getHistory(sessionId) {
    if (!sessions[sessionId]) sessions[sessionId] = [];
    return sessions[sessionId];
}

function buildLangChainHistory(rawHistory) {
    return rawHistory.map((msg) =>
        msg.role === "user"
            ? new HumanMessage(msg.content)
            : new AIMessage(msg.content)
    );
}

// POST /chat
app.post("/chat", async (req, res) => {
    const { message, sessionId = "default" } = req.body;
    if (!message) return res.status(400).json({ error: "Message is required" });

    const rawHistory = getHistory(sessionId);

    try {
        const response = await chain.invoke({
            input: message,
            history: buildLangChainHistory(rawHistory),
        });

        const reply = response.content;

        // Save to history
        rawHistory.push({ role: "user", content: message });
        rawHistory.push({ role: "assistant", content: reply });

        res.json({ reply, sessionId });
    } catch (err) {
        console.error("❌ Error:", err.message);
        res.status(500).json({ error: err.message });
    }
});

// GET /history/:sessionId
app.get("/history/:sessionId", (req, res) => {
    const history = getHistory(req.params.sessionId);
    res.json({ history });
});

// DELETE /session/:sessionId
app.delete("/session/:sessionId", (req, res) => {
    delete sessions[req.params.sessionId];
    res.json({ message: "Session cleared" });
});

// Serve frontend
app.use('/', (req, res) => {
    res.sendFile(path.join(__dirname, "public", "index.html"));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
});