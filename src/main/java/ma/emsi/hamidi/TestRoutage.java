package ma.emsi.hamidi;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.Scanner;

public class TestRoutage {

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

        DocumentParser parser = new ApacheTikaDocumentParser();
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        EmbeddingStoreWithSegments storeIA = createStore(
                "documents/rag.pdf",
                embeddingModel,
                parser
        );

        EmbeddingStoreWithSegments storeMonuments = createStore(
                "documents/MonumentsEurope.pdf",
                embeddingModel,
                parser
        );

        ContentRetriever retrieverIA = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeIA.store())
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.50)
                .build();

        ContentRetriever retrieverMonuments = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeMonuments.store())
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.50)
                .build();

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_API_KEY"))
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(
                retrieverIA,
                "Documents techniques sur l'Intelligence Artificielle, RAG, embeddings, fine-tuning."
        );
        descriptions.put(
                retrieverMonuments,
                "Documents décrivant des monuments, architecture, patrimoine culturel et historique."
        );

        LanguageModelQueryRouter queryRouter =
                new LanguageModelQueryRouter(model, descriptions);

        var augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentor)
                .build();

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("(Tapez 'q' pour quitter) \nVous : ");
            String q = scanner.nextLine();
            if (q.equalsIgnoreCase("q")) break;

            String rep = assistant.chat(q);
            System.out.println("Gemini : " + rep + "\n");
        }
    }

    private record EmbeddingStoreWithSegments(
            EmbeddingStore<TextSegment> store,
            List<TextSegment> segments
    ) {}

    private static EmbeddingStoreWithSegments createStore(
            String chemin,
            EmbeddingModel embeddingModel,
            DocumentParser parser
    ) throws Exception {

        Path path = Paths.get(chemin);
        Document doc = FileSystemDocumentLoader.loadDocument(path, parser);

        List<TextSegment> segments =
                DocumentSplitters.recursive(300, 30).split(doc);

        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        System.out.println("📄 Document chargé : " + chemin + " (" + segments.size() + " segments)");

        return new EmbeddingStoreWithSegments(store, segments);
    }
}
