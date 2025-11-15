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
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.Scanner;

public class TestNoRag {

    private static void configureLogger() {
        Logger logger = Logger.getLogger("dev.langchain4j");
        logger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        logger.addHandler(handler);
    }

    public static void main(String[] args) throws Exception {

        configureLogger();

        DocumentParser parser = new ApacheTikaDocumentParser();
        Path path = Paths.get("documents/RAG.pdf");
        Document document = FileSystemDocumentLoader.loadDocument(path, parser);

        var splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("Segments générés : " + segments.size());

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        String apiKey = System.getenv("GEMINI_API_KEY");
        if (apiKey == null) throw new RuntimeException("GEMINI_API_KEY introuvable");

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .temperature(0.2)
                .modelName("gemini-2.5-flash")
                .logRequestsAndResponses(true)
                .build();

        ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        class RouterIA implements QueryRouter {

            @Override
            public Collection<ContentRetriever> route(Query query) {

                PromptTemplate template = PromptTemplate.from(
                        "Est-ce que la requête suivante porte sur l'IA, le RAG ou le Fine-Tuning ? " +
                                "Réponds uniquement par 'oui', 'non' ou 'peut-être'.\n" +
                                "Requête : {{question}}"
                );

                String prompt = template.apply(Map.of("question", query.text())).text();

                String decision = model.chat(prompt).trim().toLowerCase();

                System.out.println("Décision du LM : " + decision);

                if (decision.contains("non")) {
                    System.out.println("Routage : Pas de RAG");
                    return Collections.emptyList();
                }

                System.out.println("Routage : RAG activé");
                return List.of(retriever);
            }
        }

        QueryRouter router = new RouterIA();

        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(augmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        System.out.println("Test conseillé :");
        System.out.println("Bonjour");
        System.out.println("Explique-moi les embeddings dans le RAG");

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("(Tapez 'q' pour quitter) \nVous : ");
            String q = scanner.nextLine();
            if (q.equalsIgnoreCase("q")) break;

            String result = assistant.chat(q);
            System.out.println("Gemini : " + result);
        }
    }
}