import os
import googleapiclient.discovery
import logging
import json
import re
from crewai_tools import BaseTool
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YouTubeCommentsTool(BaseTool):
    name: str = "YouTube Comments Fetcher"
    description: str = "Fetches all comments from a specified YouTube video using the YouTube Data API."

    def extract_video_id(self, url):
        try:
            match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
            return match.group(1) if match else None
        except Exception as e:
            logging.error(f"Video ID extract error: {e}")
            return None

    def _run(self, video_url: str) -> list:
        try:
            video_id = self.extract_video_id(video_url)
            if not video_id:
                logging.error("Invalid YouTube URL provided.")
                return []

            api_key = os.getenv("YOUTUBE_API_KEY")
            youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

            comments = []
            next_page_token = None

            while True:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=next_page_token,
                    maxResults=100,
                    textFormat="plainText"
                )
                response = request.execute()

                for item in response["items"]:
                    comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    comment = self.clean_escape_characters(comment)
                    comments.append(comment)

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

            self.save_comments_to_json(comments)
            logging.info(f"A total of {len(comments)} comments were retrieved.")
            return comments
        
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return []

    def clean_escape_characters(self, text):
        """Remove escape characters and unwanted characters from the given text."""
        text = re.sub(r'\\u[0-9A-Fa-f]{4}', '', text)  # Unicode 
        text = re.sub(r'\\U[0-9A-Fa-f]{8}', '', text)
        text = re.sub(r'\\n', ' ', text)
        text = re.sub(r'\\t', ' ', text)
        text = re.sub(r'\\r', ' ', text)
        text = re.sub(r'\\"', '', text) 
        text = re.sub(r"\\'", "'", text)
        text = re.sub(r'\\\$', '$', text)
        text = re.sub(r'\\', '', text)
        text = re.sub(r'\"', '', text) 
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def save_comments_to_json(self, comments):
        try:
            file_path = "comments.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(comments, f, ensure_ascii=False, indent=4)
            logging.info("Comments have been saved to the JSON file.")

        except Exception as e:
            logging.error(f"Failed to save comments: {e}")

class CommentsAnalysis:
    def __init__(self, comments_file_path):
        self.comments_file_path = comments_file_path
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-8b-8192"
        )
        self.results = []
        self.raw_results_file_path = "raw_results.json"
        self.final_results_file_path = "final_results.json"
        self.initialize_results_file()

    def initialize_results_file(self):
        try:
            with open(self.raw_results_file_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=4)
            logging.info(f"The file {self.raw_results_file_path} has been created.")
        except Exception as e:
            logging.error(f"Failed to create file: {e}")

    def create_agent(self) -> Agent:
        return Agent(
            role="Tech Insights Analyst",
            goal=(
                "Generate and synthesize insights from YouTube comments on tech videos. "
                "Analyze the comments from {comments_chunk} ONLY to find meaningful patterns and synthesize data into insights that highlight the core message of viewer feedback."
                "Results must be exactly in the JSON format. Do not include any extra comments, explanations, or notes beyond the analysis."
            ),
            backstory=(
                "You are a sharp-witted analyst with a passion for the tech sector. "
                "Skilled in navigating through noise in the comments data of {comments_chunk} ONLY to find meaningful patterns, "
                "you synthesize data into insights that highlight the core message of viewer feedback. "
                "Your analytical prowess ensures that the subtleties of audience engagement and preferences are captured and understood, "
                "setting the stage for informed content strategies. All your results should be grounded in evidence from the comments, so cite appropriately. "
                "Do not include any extra comments, explanations, or notes beyond the analysis."
                "Results must be exactly in the JSON format. Do not include any extra comments, explanations, or notes beyond the analysis."
                "Don't add ``` after analysis"
            ),
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
            max_iter=10
        )
    
    def create_task(self, agent: Agent, comments_chunk, task_id) -> Task:
        return Task(
            description=(
                "Analyze the comments you received and generate actionable insights for comments list {comments_chunk} ONLY. "
                "Focus on identifying common themes such as viewer pain points, requests, what viewers loved, and popular tech queries in the comments. "
                "For the requests or troubleshooting, please provide 'suggested' solutions that you might have to help me get going quicker. "
                "Add textual citation. Don't add comment numbers just comments. By citations, I mean showing the comments that led to that conclusion to support the claim. "
                "Categorize by themes like 'Requests', 'Complaints', 'Suggestions', 'Praise', 'Troubleshooting', and other relevant ones you find useful. "
                "The output should be a JSON formatted dictionary of key insights, categorized by themes like 'Requests', 'Complaints', 'Suggestions', 'Praise', 'Troubleshooting'. "
                "Each theme should contain an array of objects, each with a 'comment' and 'insight' field. "
                "Ensure the JSON output is valid by strictly following JSON format rules, such as using double quotes for strings and proper escaping of special characters. "
                "Check for missing commas, colons, and brackets to ensure the output is a valid JSON. "
                "Do NOT include any extra comments, explanations, or notes beyond the analysis. "
                "Do NOT include notes such as 'Note: The above output is a JSON formatted dictionary of key insights, categorized by themes like 'Requests', 'Complaints', 'Suggestions, and other relevant ones' "
                "or 'Please let me know if this meets the expected criteria.' "
                "Do NOT Say 'This is the final answer.' Do NOT include anything else. Just the insights. Output must be strictly a valid JSON formatted dictionary. "
                "Remove escape characters from comments. Don't include any extra comments, explanations, or notes beyond the analysis. "
                "Ensure that only the following categories are used: 'Requests', 'Complaints', 'Suggestions', 'Praise', 'Troubleshooting'. "
                "If a comment does not fit into one of these categories, classify it as 'Other'. "
                "Ensure the JSON output is valid, without trailing commas or syntax errors. Ensure proper escaping and use double quotes for strings."
                "Don't mention that ```json to by the end of the task do not do that. Don't include any extra comments, explanations, or notes beyond the analysis. Don't include any comment numbers."
                "Don't add ``` after analysis"
                 "Results must be exactly in the JSON format. Do not include any extra comments, explanations, or notes beyond the analysis."
            ),
            agent=agent,
            expected_output=(
                "A JSON formatted dictionary of key insights, categorized by themes like 'Requests', 'Complaints', 'Suggestions', 'Praise', 'Troubleshooting'. "
                "Each theme should contain an array of objects, each with a 'comment' and 'insight' field. "
                "Each result must be in exactly the same format. Do NOT ADD any other NOTES, COMMENTS, or EXPLANATIONS. Output must be strictly JSON formatted dictionary. "
                "Do NOT Say 'This is the final answer.' Do NOT add any extra data to your results. Use double quotes for all strings. Don't include any comment numbers."
                "Ensure the JSON output is valid, without trailing commas or syntax errors. Ensure proper escaping and use double quotes for strings."
                 "Results must be exactly in the JSON format. Do not include any extra comments, explanations, or notes beyond the analysis."
            ),
            inputs={"comments_chunk": comments_chunk}
        )

    
    def create_crew(self, agent: Agent, tasks: list) -> Crew:
        return Crew(
            agents=[agent],
            tasks=tasks,
            process=Process.sequential,
            verbose=2,
            memory=False,
            max_rpm=100
        )
    
    def split_comments(self, comments, chunk_size):
        """Splits comments into chunks of a specified size."""
        return [comments[i:i + chunk_size] for i in range(0, len(comments), chunk_size)]
    
    def append_result_to_json(self, result):
        """Appends the result to the raw_results.json file."""
        try:
            with open(self.raw_results_file_path, "r+", encoding="utf-8") as f:
                results = json.load(f)
                try:
                    fixed_result = self.fix_trailing_commas(result)
                    json_result = json.loads(fixed_result)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error: {e}")
                    logging.error(f"Problematic result: {result}")
                    return

                results.append(json_result)

                f.seek(0)
                json.dump(results, f, ensure_ascii=False, indent=4)
            logging.info("The result has been appended to the raw_results.json file.")

        except Exception as e:
            logging.error(f"Failed to append the result: {e}")

    def fix_trailing_commas(self, json_str):
        """Fix trailing commas in JSON string."""
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = re.sub(r',\s*}', '}', json_str)
        return json_str

    def merge_results(self):
        """Merges all comments into a single JSON structure."""

        try:
            with open(self.raw_results_file_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            
            final_results = {
                "Requests": [],
                "Complaints": [],
                "Suggestions": [],
                "Praise": [],
                "Troubleshooting": [],
                "Other": []
            }
            
            for entry in all_results:
                if isinstance(entry, dict):
                    for key, value in entry.items():
                        if key in final_results and isinstance(value, list):
                            final_results[key].extend(value)
                        else:
                            final_results["Other"].extend(value)
                else:
                    logging.error(f"Unexpected entry type: {type(entry)} - {entry}")

            with open(self.final_results_file_path, "w", encoding="utf-8") as f:
                json.dump(final_results, f, ensure_ascii=False, indent=4)
            
            logging.info(f"All results have been merged into the {self.final_results_file_path} file.")

        except Exception as e:
            logging.error(f"Failed to merge results: {e}")

    def run(self):
        try:
            if not os.path.exists(self.comments_file_path):
                raise FileNotFoundError(f"Comments file not found: {self.comments_file_path}")
            
            with open(self.comments_file_path, 'r', encoding='utf-8') as file:
                comments_data = json.load(file)
            
            if comments_data is None:
                logging.error("Failed to load comments data.")
                return
            
            chunk_size = 20
            comments_chunks = self.split_comments(comments_data, chunk_size)
            
            logging.info(f"Loaded comments: {comments_data[:5]}")
            
            for idx, chunk in enumerate(comments_chunks):
                agent = self.create_agent()
                task = self.create_task(agent, chunk, idx + 1)
                crew = self.create_crew(agent, [task])
                
                result = crew.kickoff(inputs={"comments_chunk": chunk})
                
                logging.debug(f"Result type: {type(result)}")
                logging.debug(f"Result content: {result}")

                if isinstance(result, str):
                    fixed_result = self.fix_trailing_commas(result)
                else:
                    fixed_result = self.fix_trailing_commas(str(result))  # or result.get_output() if such a method exists

                logging.info(f"Task {idx + 1} raw result: {fixed_result}")
                
                try:
                    json_result = json.loads(fixed_result)
                    self.append_result_to_json(fixed_result)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error after validation: {e}")
                    logging.error(f"Problematic result: {fixed_result}")
                
                logging.info(f"Task {idx + 1} output: {fixed_result}")

            self.merge_results()

        except Exception as e:
            logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    yt_tool = YouTubeCommentsTool()
    video_url = input("🚀 Enter YouTube URL: ")
    yt_tool._run(video_url)

    comments_file_path = "comments.json"
    comments_analysis = CommentsAnalysis(comments_file_path)
    comments_analysis.run()
