// app/(tabs)/chat.tsx
import { useState, useRef } from "react";
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  FlatList,
  ActivityIndicator,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { KeyboardStickyView } from "react-native-keyboard-controller";
import { sendChat, ChatMessage, ChatResponse } from "../../lib/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

function MarkdownText({ content }: { content: string }) {
  const lines = content.split("\n");
  let inCodeBlock = false;

  function renderInline(text: string) {
    return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, index) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return (
          <Text key={index} style={styles.boldText}>
            {part.slice(2, -2)}
          </Text>
        );
      }
      if (part.startsWith("`") && part.endsWith("`")) {
        return (
          <Text key={index} style={styles.inlineCodeText}>
            {part.slice(1, -1)}
          </Text>
        );
      }
      return <Text key={index}>{part.replace(/\*/g, "")}</Text>;
    });
  }

  return (
    <Text style={styles.messageText}>
      {lines.map((line, index) => {
        if (line.trim().startsWith("```")) {
          inCodeBlock = !inCodeBlock;
          return index < lines.length - 1 ? "\n" : null;
        }

        if (inCodeBlock) {
          return (
            <Text key={index} style={styles.codeText}>
              {line}
              {index < lines.length - 1 ? "\n" : ""}
            </Text>
          );
        }

        const bullet = line.match(/^\s*[*-]\s+(.*)$/);
        const numbered = line.match(/^\s*(\d+)\.\s+(.*)$/);
        const contentLine = bullet ? bullet[1] : numbered ? numbered[2] : line;
        const prefix = bullet ? "•  " : numbered ? `${numbered[1]}.  ` : "";

        return (
          <Text key={index}>
            {prefix}
            {renderInline(contentLine)}
            {index < lines.length - 1 ? "\n" : ""}
          </Text>
        );
      })}
    </Text>
  );
}

export default function ChatScreen() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Hi! I'm AeroGuard AI. Ask me anything about your air quality data — current readings, trends, health advice, and more.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const flatListRef = useRef<FlatList>(null);

  async function handleSend() {
    const question = input.trim();
    if (!question || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: question,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const history: ChatMessage[] = messages
        .filter((m) => m.id !== "welcome")
        .map((m) => ({ role: m.role, content: m.content }));

      const response: ChatResponse = await sendChat(question, history);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.answer,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `⚠️ Error: ${err.message || "Could not reach the AI assistant."}`,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
      setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 100);
    }
  }

  function renderMessage({ item }: { item: Message }) {
    const isUser = item.role === "user";
    return (
      <View
        style={[
          styles.messageBubble,
          isUser ? styles.userBubble : styles.assistantBubble,
        ]}
      >
        {isUser ? (
          <Text style={[styles.messageText, styles.userMessageText]}>
            {item.content}
          </Text>
        ) : (
          <MarkdownText content={item.content} />
        )}
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={["bottom"]}>
      <View style={styles.container}>
        <FlatList
          ref={flatListRef}
          data={messages}
          keyExtractor={(item) => item.id}
          renderItem={renderMessage}
          contentContainerStyle={styles.messagesContainer}
          onContentSizeChange={() =>
            flatListRef.current?.scrollToEnd({ animated: true })
          }
        />

        {loading && (
          <View style={styles.loadingIndicator}>
            <ActivityIndicator size="small" color="#00A699" />
            <Text style={styles.loadingText}>AeroGuard AI is thinking...</Text>
          </View>
        )}

        <KeyboardStickyView>
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.input}
              value={input}
              onChangeText={setInput}
              placeholder="Ask about air quality..."
              placeholderTextColor="#999"
              multiline
              maxLength={500}
              onSubmitEditing={handleSend}
            />
            <TouchableOpacity
              style={[styles.sendButton, (!input.trim() || loading) && styles.sendButtonDisabled]}
              onPress={handleSend}
              disabled={!input.trim() || loading}
            >
              <Text style={styles.sendButtonText}>Send</Text>
            </TouchableOpacity>
          </View>
        </KeyboardStickyView>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f5f5f5",
  },
  messagesContainer: {
    padding: 16,
    paddingBottom: 8,
  },
  messageBubble: {
    maxWidth: "80%",
    borderRadius: 16,
    padding: 12,
    marginVertical: 4,
  },
  userBubble: {
    alignSelf: "flex-end",
    backgroundColor: "#00A699",
    borderBottomRightRadius: 4,
  },
  assistantBubble: {
    alignSelf: "flex-start",
    backgroundColor: "#fff",
    borderBottomLeftRadius: 4,
    borderWidth: 1,
    borderColor: "#e0e0e0",
  },
  messageText: {
    fontSize: 15,
    color: "#333",
    lineHeight: 20,
  },
  boldText: {
    fontWeight: "700",
  },
  inlineCodeText: {
    fontFamily: "Menlo",
    backgroundColor: "#eeeeee",
  },
  codeText: {
    fontFamily: "Menlo",
    fontSize: 13,
    lineHeight: 18,
    color: "#1f2937",
    backgroundColor: "#eeeeee",
  },
  userMessageText: {
    color: "#fff",
  },
  loadingIndicator: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 8,
  },
  loadingText: {
    marginLeft: 8,
    fontSize: 13,
    color: "#666",
  },
  inputContainer: {
    flexDirection: "row",
    alignItems: "flex-end",
    padding: 12,
    backgroundColor: "#fff",
    borderTopWidth: 1,
    borderTopColor: "#e0e0e0",
  },
  input: {
    flex: 1,
    maxHeight: 100,
    backgroundColor: "#f0f0f0",
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 10,
    fontSize: 15,
    color: "#333",
  },
  sendButton: {
    marginLeft: 10,
    backgroundColor: "#00A699",
    borderRadius: 20,
    paddingHorizontal: 18,
    paddingVertical: 10,
  },
  sendButtonDisabled: {
    backgroundColor: "#ccc",
  },
  sendButtonText: {
    color: "#fff",
    fontWeight: "600",
    fontSize: 15,
  },
});
