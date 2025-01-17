import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { StateGraph } from "@langchain/langgraph";
import { Annotation } from "@langchain/langgraph";
import { tool,StructuredTool } from "@langchain/core/tools";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { MongoClient } from "mongodb";
import { z } from "zod";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import "dotenv/config";

export async function callAgent(client: MongoClient, query: string, thread_id: string) {
    // Define the MongoDB database and collection
    const dbName = "events_database";
    const db = client.db(dbName);
    const collection = db.collection("events");
  
    // Define the graph state
    const GraphState = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: (x, y) => x.concat(y),
      }),
    });

 
  
    // Define the tools for the agent to use
    const eventLookupTool = tool(
      async ({ query, n = 10 }) => {
        console.log("Event lookup tool called");
  
        const dbConfig = {
          collection: collection,
          indexName: "vector_index",
          textKey: "embedding_text",
          embeddingKey: "embedding",
        };
  
        // Initialize vector store
        const vectorStore = new MongoDBAtlasVectorSearch(
          new OpenAIEmbeddings(),
          dbConfig
        );
  
        const result = await vectorStore.similaritySearchWithScore(query, n);
        return JSON.stringify(result);
      },
      {
        name: "events_lookup",
        description: "Gathers events information from Events database",
        schema: z.object({
          query: z.string().describe("The search query"),
          n: z
            .number()
            .optional()
            .default(10)
            .describe("Number of results to return"),
        }),
      }
    );
  
    const tools = [eventLookupTool];
    
    // We can extract the state typing via `GraphState.State`
    const toolNode = new ToolNode<typeof GraphState.State>(tools);
  
    const model = new ChatAnthropic({
      model: "claude-3-5-sonnet-20240620",
      temperature: 0,
    }).bindTools(tools);
  
    // Define the function that determines whether to continue or not
    function shouldContinue(state: typeof GraphState.State) {
      const messages = state.messages;
      const lastMessage = messages[messages.length - 1] as AIMessage;
  
      // If the LLM makes a tool call, then we route to the "tools" node
      if (lastMessage.tool_calls?.length) {
        return "tools";
      }
      // Otherwise, we stop (reply to the user)
      return "__end__";
    }
  
    // Define the function that calls the model
    async function callModel(state: typeof GraphState.State) {
      const prompt = ChatPromptTemplate.fromMessages([
        [
          "system",
          `You are a helpful AI assistant/tour guide specializing in black events & restaurants, clubs, day parties, brunches, dinner parties and more around the world although your main focus is on black events and clubs/venues, 
          collaborating with other assistants. Use the provided tools to progress towards answering the question.
           If the user asks you for recommendations, provide a list that matches what they're looking for. 
           If a user asks for recommendations, provide anything thats happening within the week but if there isn't anything give alternative recommendations similar to their query or alternative places to check out.
           If the user asks you to create an itinerary using the information, in this case provide one.
           If you are asked about specific events, clubs, venues, restaurants in your database provide this information to the user regardless of them specifying a city, then ask if they are interested in similar events.
           If the user doesn't specify a city they will be in and they don't ask about a specific event in the database, ask them what city they would like the recommendations for before providing a recommendation.
           If you are unable to fully answer, that's OK, another assistant with different tools will help where you left off. Execute what you can to make progress. 
           If you or any of the other assistants have the final answer or deliverable, prefix your response with RECOMMENDATIONS so the team knows to stop. 
           If you are asked about restaurants, you can provide recommendations from your tools as these are listed in there.
           for each recommendation you provide or when you provide information for a specific event, club, restaurant etc. convert the instagram handle to a clickable link and provide this as well
           You have access to the following tools: {tool_names}.\n{system_message}\nCurrent time: {time}.`,
        ],
        new MessagesPlaceholder("messages"),
      ]);
  
      const formattedPrompt = await prompt.formatMessages({
        system_message: "You are helpful Black events tour guide/Chatbot Agent.",
        time: new Date().toISOString(),
        tool_names: tools.map((tool) => tool.name).join(", "),
        messages: state.messages,
      });
  
      const result = await model.invoke(formattedPrompt);
  
      return { messages: [result] };
    }
  
    // Define a new graph
    const workflow = new StateGraph(GraphState)
      .addNode("agent", callModel)
      .addNode("tools", toolNode)
      .addEdge("__start__", "agent")
      .addConditionalEdges("agent", shouldContinue)
      .addEdge("tools", "agent");
  
    // Initialize the MongoDB memory to persist state between graph runs
    const checkpointer = new MongoDBSaver({ client, dbName });
  
    // This compiles it into a LangChain Runnable.
    // Note that we're passing the memory when compiling the graph
    const app = workflow.compile({ checkpointer });
  
    // Use the Runnable
    const finalState = await app.invoke(
      {
        messages: [new HumanMessage(query)],
      },
      { recursionLimit: 15, configurable: { thread_id: thread_id } }
    );
  
    // console.log(JSON.stringify(finalState.messages, null, 2));
    console.log(finalState.messages[finalState.messages.length - 1].content);
  
    return finalState.messages[finalState.messages.length - 1].content;
  }