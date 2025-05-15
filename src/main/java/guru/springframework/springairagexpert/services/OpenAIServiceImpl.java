package guru.springframework.springairagexpert.services;

import guru.springframework.springairagexpert.model.Answer;
import guru.springframework.springairagexpert.model.Question;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.chat.prompt.SystemPromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;
import org.springframework.web.client.ResourceAccessException;

import java.util.List;
import java.util.Map;


@Slf4j
@RequiredArgsConstructor
@Service
public class OpenAIServiceImpl implements OpenAIService {

    final ChatModel chatModel;
    final VectorStore vectorStore;

    @Value("classpath:/templates/rag-prompt-template.st")
    private Resource ragPromptTemplate;

    @Value("classpath:/templates/system-message.st")
    private Resource systemMessageTemplate;

    @Override
    public Answer getAnswer(Question question) {
        try {
            PromptTemplate systemMessagePromptTemplate = new SystemPromptTemplate(systemMessageTemplate);
            Message systemMessage = systemMessagePromptTemplate.createMessage();

            List<Document> documents = vectorStore.similaritySearch(SearchRequest.builder()
                    .query(question.question()).topK(5).build());
            List<String> contentList = documents.stream().map(Document::getContent).toList();

            PromptTemplate promptTemplate = new PromptTemplate(ragPromptTemplate);
            Message userMessage = promptTemplate.createMessage(Map.of("input", question.question(), "documents",
                    String.join("\n", contentList)));

            ChatResponse response = chatModel.call(new Prompt(List.of(systemMessage, userMessage)));

            return new Answer(response.getResult().getOutput().getContent());
        } catch (ResourceAccessException e) {
            log.error("Failed to connect to OpenAI service", e);
            return new Answer("Sorry, I'm currently unable to connect to the AI service. Please try again later.");
        } catch (Exception e) {
            log.error("Error processing question", e);
            return new Answer("An error occurred while processing your question. Please try again later.");
        }
    }

}











