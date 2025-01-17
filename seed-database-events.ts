import { MongoClient } from "mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { z } from "zod";
import "dotenv/config";
import csv from 'csv-parser';
import fs from 'fs';
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import "dotenv/config";

const client = new MongoClient(process.env.MONGODB_ATLAS_URI as string);

const llm = new ChatOpenAI({
    modelName: "gpt-4o-mini",
    temperature: 0.7,
  });

const EventSchema = z.object({
  name: z.string(),
  type: z.string(),
  date: z.string(),
  time: z.string(),
  location: z.string(),
  instagram: z.string(),
  city: z.string(),
  notes: z.string(),
  similarTo: z.string(),
});

type Event = z.infer<typeof EventSchema>;

async function parseCSVFile(filePath: string): Promise<Event[]> {
  return new Promise((resolve, reject) => {
    const events: Event[] = [];
    fs.createReadStream(filePath)
      .pipe(csv())
      .on('data', (data:Event) => {
        // Map CSV data to EventSchema
        const event: Event = {
          name: data.name,
          type: data.type,
          date: data.date,
          time: data.time,
          location: data.location,
          instagram: data.instagram,
          city: data.city,
          notes: data.notes,
          similarTo: data.similarTo,
        };
        events.push(event);
      })
      .on('end', () => {
        resolve(events);
      })
      .on('error', (err:any) => {
        reject(err);
      });
  });
}

async function createEventSummary(event: Event): Promise<string> {
    return new Promise((resolve) => {
      const name = `${event.name}`;
      const type = `${event.type}`;
      const date = `${event.date}`;
      const time = `${event.time}`;
      const location = `${event.location}`;
      const instagram = `${event.instagram}`;
      const city = `${event.city}`;
      const notes = event.notes;
      const similarTo = `${event.similarTo}`;
  
      const summary = `Name: ${name}. Type: ${type}. Date: ${date}. Time: ${time}. Location: ${location}. Instagram: ${instagram}. City: ${city}. Notes: ${notes}. Similar To ${similarTo}`;
  
      resolve(summary);
    });
  }


async function embedEventsWithVectorSearch(events: Event[]): Promise<void> {
  try {
    await client.connect();
    await client.db("admin").command({ ping: 1 });
    console.log("Pinged your deployment. You successfully connected to MongoDB!");

    const db = client.db("events_database");
    const collection = db.collection("events");

    // Prepare the records with summaries for vector search
    const recordsWithSummaries = await Promise.all(
      events.map(async (record) => ({
        pageContent: await createEventSummary(record),
        metadata: {...record},
      }))
    );

    // Use vector search to store events
    for (const record of recordsWithSummaries) {
      await MongoDBAtlasVectorSearch.fromDocuments(
        [record],
        new OpenAIEmbeddings(),
        {
          collection,
          indexName: "vector_index",
          textKey: "embedding_text",
          embeddingKey: "embedding",
        }
      );
      console.log("Successfully processed & saved event:", record.metadata.name);
    }

    console.log("Database seeding with vector search completed.");
  } catch (error) {
    console.error("Error seeding database with vector search:", error);
  } finally {
    await client.close();
  }
}

async function main() {
  try {
    const filePath = './events-db.csv';  
    const events = await parseCSVFile(filePath);

    await embedEventsWithVectorSearch(events);  // Insert events with vector search
  } catch (error) {
    console.error("Error processing events:", error);
  }
}

main().catch(console.error);