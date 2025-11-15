package ma.emsi.hamidi;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

public class RagNaif {

    public static void main(String[] args) throws Exception {

        DocumentParser documentParser = new ApacheTikaDocumentParser();

        Path path = Paths.get("documents/RAG.pdf");
        Document document = FileSystemDocumentLoader.loadDocument(path, documentParser);

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("Nombre de segments : " + segments.size());

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        System.out.println("Nombre d'embeddings générés : " + embeddings.size());

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);
        System.out.println("Enregistrement des embeddings terminé avec succès !");

        String cle = System.getenv("GEMINI_API_KEY");
        if (cle == null) {
            throw new IllegalStateException("Variable d'environnement GEMINI_API_KEY invalide !");
        }

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(cle)
                .temperature(0.3)
                .modelName("gemini-2.5-flash")
                .build();

        EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        var memory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(memory)
                .contentRetriever(retriever)
                .build();

        try (Scanner scanner = new Scanner(System.in)) {
            System.out.println("Posez votre question :");
            while (true) {
                System.out.print("(Tapez 'q' pour quitter) Vous : ");
                String question = scanner.nextLine();
                if (question.equalsIgnoreCase("q"))
                    break;
                String reponse = assistant.chat(question);
                System.out.println("Gemini : " + reponse);
            }
        }
    }
}