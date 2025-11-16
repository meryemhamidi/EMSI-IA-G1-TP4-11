package ma.emsi.hamidi;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestWebSearch {

    private static void configureLogger() {
        Logger logger = Logger.getLogger("dev.langchain4j");
        logger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        logger.addHandler(handler);

        System.out.println("Logger activé");
    }

    public static void main(String[] args) throws Exception {

        configureLogger();

        System.out.println("📄 Chargement du PDF...");

        DocumentParser parser = new ApacheTikaDocumentParser();
        Path path = Paths.get("documents/RAG.pdf");

        Document document = FileSystemDocumentLoader.loadDocument(path, parser);
        List<TextSegment> segments = DocumentSplitters.recursive(300, 30).split(document);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        System.out.println("PDF indexé : " + segments.size() + " segments");

        System.out.println("Initialisation du modèle Gemini...");

        String GEMINI_API_KEY = System.getenv("GEMINI_API_KEY");
        if (GEMINI_API_KEY == null) throw new IllegalStateException("GEMINI_API_KEY introuvable !");

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(GEMINI_API_KEY)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        ContentRetriever retrieverLocal = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.5)
                .build();

        System.out.println("Initialisation du moteur de recherche Tavily...");

        String TAVILY_API_KEY = System.getenv("TAVILY_API_KEY");
        if (TAVILY_API_KEY == null) throw new IllegalStateException("TAVILY_API_KEY introuvable !");

        TavilyWebSearchEngine tavilyEngine = TavilyWebSearchEngine.builder()
                .apiKey(TAVILY_API_KEY)
                .build();

        ContentRetriever retrieverWeb = WebSearchContentRetriever.builder()
                .webSearchEngine(tavilyEngine)
                .maxResults(3)
                .build();

        System.out.println("Configuration du QueryRouter...");

        QueryRouter router = new DefaultQueryRouter(List.of(
                retrieverLocal,
                retrieverWeb
        ));

        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(augmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.print("(Tapez 'q' pour quitter) \nVous : ");
                String question = scanner.nextLine();
                if (question.equalsIgnoreCase("q")) break;

                String response = assistant.chat(question);
                System.out.println("Gemini : " + response);
            }
        }
    }
}
