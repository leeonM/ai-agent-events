import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { StateGraph } from "@langchain/langgraph";
import { Annotation } from "@langchain/langgraph";
import { tool, StructuredTool } from "@langchain/core/tools";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { MongoClient } from "mongodb";
import { z } from "zod";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import "dotenv/config";

// First Agent: MongoDB Event Lookup
export async function eventLookupAgent(client: MongoClient, query: string, thread_id: string) {
    try {
        const dbName = "events_database";
        const db = client.db(dbName);
        const collection = db.collection("events");
      
        const GraphState = Annotation.Root({
          messages: Annotation<BaseMessage[]>({
            reducer: (x, y) => x.concat(y),
          }),
        });

        const eventLookupTool = tool(
          async ({ query, n = 10 }) => {
            try {
                console.log("Event lookup tool called with query:", query);
      
                const dbConfig = {
                  collection: collection,
                  indexName: "vector_index",
                  textKey: "embedding_text",
                  embeddingKey: "embedding",
                };
      
                const vectorStore = new MongoDBAtlasVectorSearch(
                  new OpenAIEmbeddings(),
                  dbConfig
                );
      
                const result = await vectorStore.similaritySearchWithScore(query, n);
                
                if (!result || result.length === 0) {
                    return JSON.stringify({ message: "NO_RESULTS_FOUND" });
                }
                
                return JSON.stringify(result);
            } catch (error:any) {
                console.error("Error in event lookup tool:", error);
                return JSON.stringify({ message: "NO_RESULTS_FOUND", error: error.message });
            }
          },
          {
            name: "events_lookup",
            description: "Searches for events information in the Events database",
            schema: z.object({
              query: z.string().describe("The search query"),
              n: z.number().optional().default(10).describe("Number of results to return"),
            }),
          }
        );

        const tools = [eventLookupTool];
        const toolNode = new ToolNode<typeof GraphState.State>(tools);
      
        const model = new ChatAnthropic({
          model: "claude-3-5-sonnet-20240620",
          temperature: 0,
        }).bindTools(tools);
      
        function shouldContinue(state: typeof GraphState.State) {
          const messages = state.messages;
          const lastMessage = messages[messages.length - 1] as AIMessage;
          return lastMessage.tool_calls?.length ? "tools" : "__end__";
        }
      
        async function callModel(state: typeof GraphState.State) {
          try {
              const prompt = ChatPromptTemplate.fromMessages([
                [
                  "system",
                  `You are a helpful AI assistant/tour guide specializing in black events & restaurants, clubs, day parties, brunches, dinner parties and more around the world with a focus on black events and venues.

                  ALWAYS follow this process:
                  1. Use the events_lookup tool to search the database
                  2. After getting results, ALWAYS format your response as follows:
                     
                     If results found:
                     [Detailed list of database results]
                     
                     PASS_TO_WEB_AGENT
                     Search for: [List each venue, event, or detail that needs verification]

                     If no results:
                     NO_RESULTS_FOUND
                     
                  Key requirements:
                  - ALWAYS include either PASS_TO_WEB_AGENT or NO_RESULTS_FOUND in your response
                  - If you find any results, always pass them to the web agent for verification
                  - Include Instagram handles for each venue/event
                  - If the user doesn't specify a city, ask for their preferred location
                  - For recommendations, focus on events within the week
                  - If no current events, suggest similar alternatives

                  Remember: The web agent will verify and enhance your results, so ALWAYS pass your findings forward with PASS_TO_WEB_AGENT.

                  You have access to the following tools: {tool_names}.\n{system_message}\nCurrent time: {time}.`,
                ],
                new MessagesPlaceholder("messages"),
              ]);
          
              const formattedPrompt = await prompt.formatMessages({
                system_message: "You are the Database Search Agent.",
                time: new Date().toISOString(),
                tool_names: tools.map((tool) => tool.name).join(", "),
                messages: state.messages,
              });
          
              const result = await model.invoke(formattedPrompt);
              return { messages: [result] };
          } catch (error) {
              console.error("Error in model call:", error);
              throw error;
          }
        }
      
        const workflow = new StateGraph(GraphState)
          .addNode("agent", callModel)
          .addNode("tools", toolNode)
          .addEdge("__start__", "agent")
          .addConditionalEdges("agent", shouldContinue)
          .addEdge("tools", "agent");
      
        const checkpointer = new MongoDBSaver({ client, dbName });
        const app = workflow.compile({ checkpointer });
      
        const finalState = await app.invoke(
          {
            messages: [new HumanMessage(query)],
          },
          { 
              recursionLimit: 15, 
              configurable: { thread_id: thread_id },
              timeout: 60000 // 30 second timeout
          }
        );
      
        return finalState.messages[finalState.messages.length - 1].content;
    } catch (error) {
        console.error("Error in event lookup agent:", error);
        return "NO_RESULTS_FOUND";
    }
}

// Second Agent: Web Search and Final Recommendations

// Simplified Web Search Agent
async function webSearchAgent(query: string, dbResults: string, thread_id: string) {
    const searchResults:any = [];
    const model = new ChatAnthropic({
        model: "claude-3-5-sonnet-20240620",
        temperature: 0,
    });

    const tavilyTool = new TavilySearchResults({
        maxResults: 3,
    });

    try {
        // Parse the enhanced query if available
        let searchQueries = [];
        try {
            const parsedQuery = JSON.parse(query);
            if (parsedQuery.searchSummary) {
                // Extract individual search terms from the summary
                searchQueries = parsedQuery.searchSummary
                    .split('\n')
                                    // @ts-ignore
                    .map(line => line.trim())
                                    // @ts-ignore
                    .filter(line => line.length > 0);
            }
        } catch {
            searchQueries = [query];
        }

        // Perform searches with individual timeouts
        for (const searchQuery of searchQueries) {
            console.log("Web search tool called with query:", searchQuery);
            console.log("Waiting for search results...");
            
            try {
                const searchPromise = tavilyTool.invoke(searchQuery);
                const timeoutPromise = new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('Search timeout')), 10000)
                );
                
                const result = await Promise.race([searchPromise, timeoutPromise]);
                console.log("Search completed successfully");
                searchResults.push({ query: searchQuery, results: result });
            } catch (error:any) {
                console.log("Search failed or timed out:", error.message);
                searchResults.push({ query: searchQuery, error: error.message });
            }
        }

        // Process results with a timeout
        const processPromise = new Promise(async (resolve) => {
            try {
                // Format the collected results

                const resultsContext = `
                Original Query: ${query}
                Database Results: ${dbResults} 
                Web Search Results: ${JSON.stringify(searchResults, null, 2)}
                ` 

                const response = await model.invoke(
                    `You are an event recommendation assistant. Based on these search results, provide a concise summary of verified events and venues. Include social media links and website URLs where available. Start your response with RECOMMENDATIONS:\n\n${resultsContext}`
                );
                
                resolve(response.content);
            } catch (error) {
                resolve(`RECOMMENDATIONS\nI found these events but couldn't verify all details:\n${dbResults}`);
            }
        });

        const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Processing timeout')), 15000)
        );

        return await Promise.race([processPromise, timeoutPromise])
            .catch(() => `RECOMMENDATIONS\nI found these events but couldn't verify all details:\n${dbResults}`);

    } catch (error) {
        console.error("Error in web search agent:", error);
        return `RECOMMENDATIONS\nI found these events but couldn't verify all details:\n${dbResults}`;
    }
}

export async function callAgent(client: MongoClient, query: string, thread_id: string) {
    try {
        console.log("Starting agent workflow with query:", query);
        
        // First, call the event lookup agent
        const dbResults = await eventLookupAgent(client, query, thread_id);
        console.log("Database results received");
        
        // Check if we should proceed to web search
        if (dbResults && typeof dbResults === 'string') {
            if (dbResults.includes("PASS_TO_WEB_AGENT")) {
                console.log("Proceeding to web search");
                const searchSummary = dbResults.split("PASS_TO_WEB_AGENT")[1].trim();
                
                const enhancedQuery = {
                    originalQuery: query,
                    searchSummary: searchSummary,
                    dbResults: dbResults.split("PASS_TO_WEB_AGENT")[0].trim()
                };

                // Set a timeout for the entire web search process
                try {
                    const webSearchPromise = webSearchAgent(JSON.stringify(enhancedQuery), dbResults, thread_id);
                    const timeoutPromise = new Promise((_, reject) => 
                        setTimeout(() => reject(new Error('Web search process timeout')), 45000)
                    );
                    
                    const finalResults = await Promise.race([webSearchPromise, timeoutPromise]);
                    return finalResults;
                } catch (error) {
                    console.log("Web search timed out or failed, returning available information");
                    return `RECOMMENDATIONS\nI found some events but couldn't verify all details:\n${dbResults.split("PASS_TO_WEB_AGENT")[0]}`;
                }
            } else if (dbResults.includes("NO_RESULTS_FOUND")) {
                console.log("No database results, proceeding with web-only search");
                try {
                    const webSearchPromise = webSearchAgent(query, "No database results found.", thread_id);
                    const timeoutPromise = new Promise((_, reject) => 
                        setTimeout(() => reject(new Error('Web search process timeout')), 45000)
                    );
                    
                    const finalResults = await Promise.race([webSearchPromise, timeoutPromise]);
                    return finalResults;
                } catch (error) {
                    return "RECOMMENDATIONS\nI couldn't find any specific events in our database or through web search. Please try a different city or check back later.";
                }
            }
        }
        
        console.log("Returning database results only");
        return dbResults;
    } catch (error) {
        console.error("Error in main agent workflow:", error);
        return "I apologize, but I encountered an error while processing your request. Please try again or rephrase your query.";
    }
}