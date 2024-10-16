from typing import List, Tuple, Optional
import tiktoken
from tqdm import tqdm
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get('API_KEY'))

def fix_summary(paragraph: str):
    prompt = (
        '''
        You are a highly-intellectual college student and you are reading the provided material to create a comprehensive summary notes for your upcoming test.
            Generate a summary notes that covers the whole provided material. It should be clear, concise, and easy to understand.
            The summary notes should follow these formats: Introduction/Main Topic, The Sub-topics/Sub-Sections, Key Points of each Sub-topics/Sub-Sections, and lastly Overview Summary"
            Introduction/Main Topic Format: should be two to three sentences long. Should contain a concise background/context of what the whole note summary revolves around. Use an Abstractive Summary which you can rephrase the original texts to be easily understood.
            Sub-Topic/Sub-Sections Format: should be organized sections about specific ideas relating to the Main Topic. Give a one sentence explanation about this section. Use Extractive Summary (Do not rephrase, change, or replace any words) if there's an explanation or brief description about the Sub-Topic that can be found on the material. Use Abstractive Method (paraphrase) if there is none.
            Key Points Format: these are points that are inside on each Sub-Topic/Sub-Sections. For clarity use bullet points if the order of the key points doesn't matter and use numbered list if the order of the key points matter. Strictly use Extractive Summary (Do not rephrase, change, or replace any words) in creating key points.
            Overview Summary: should be one paragraph and contains final thoughts that encapsulates the entire summary note. Mentioning the purpose of the topic. Use an Abstractive Summary on this, use Layman's Term.
            Highlight the important words, or phrase by making it bold or use annotations to highlight importance of it.


            Example Format of the Note Summary:

            Introduction
            The Renaissance was a transformative period in European history, marking a cultural, artistic, and intellectual revival.
            Spanning roughly from the 14th to the 17th century, it emphasized the rediscovery of classical knowledge and human potential...

            Key Points
            The Origins of the Renaissance
            This section explores the historical conditions that led to the Renaissance's emergence.
            - Florence, Italy: the place where the Renaissance's began at late 14th century.
            - Scholars rediscovered ancient Greek and Roman texts.
            ...

            The Development of the Printing Press
            This section explains the steps involved in the invention and spread of the printing press, which revolutionized the distribution of knowledge.
            1. 1440: Johannes Gutenberg invents the movable-type printing press in Germany.
            2. 1455: Gutenberg prints the first major book, the Gutenberg Bible.
            ...

            Overview Summary:
            The Renaissance was a profound period of change, sparking shifts in thought, art, and science. It laid the groundwork for modern education, creative expression,
            and technological advancements, illustrating the enduring influence of human curiosity and learning on societal progress...

	        RESPOND ONLY in html tags format use ordered list, headings, and unordered list only, no need for a full html body structure use tags directly. Remove extras like ``` or the tag/word html and do not use markdowns elements. You may also read inline LaTex and block LaTex formulas

        '''
        )

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": prompt },
            {"role": "user", "content": paragraph},
        ],
        temperature=0.7,
    )
    
    title = response.choices[0].message.content
    

    return title

def get_chat_completion(messages, model='gpt-3.5-turbo-1106'):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

def tokenize(text: str) -> List[str]:
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-1106')
    return encoding.encode(text)


# This function chunks a text into smaller pieces based on a maximum token count and a delimiter.
def chunk_on_delimiter(input_string: str,
                       max_tokens: int, delimiter: str) -> List[str]:
    chunks = input_string.split(delimiter)
    combined_chunks, _, dropped_chunk_count = combine_chunks_with_no_minimum(
        chunks, max_tokens, chunk_delimiter=delimiter, add_ellipsis_for_overflow=True
    )
    if dropped_chunk_count > 0:
        print(f"warning: {dropped_chunk_count} chunks were dropped due to overflow")
    combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]
    return combined_chunks


# This function combines text chunks into larger blocks without exceeding a specified token count. It returns the combined text blocks, their original indices, and the count of chunks dropped due to overflow.
def combine_chunks_with_no_minimum(
        chunks: List[str],
        max_tokens: int,
        chunk_delimiter="\n\n",
        header: Optional[str] = None,
        add_ellipsis_for_overflow=False,
) -> Tuple[List[str], List[int]]:
    dropped_chunk_count = 0
    output = []  # list to hold the final combined chunks
    output_indices = []  # list to hold the indices of the final combined chunks
    candidate = (
        [] if header is None else [header]
    )  # list to hold the current combined chunk candidate
    candidate_indices = []
    for chunk_i, chunk in enumerate(chunks):
        chunk_with_header = [chunk] if header is None else [header, chunk]
        if len(tokenize(chunk_delimiter.join(chunk_with_header))) > max_tokens:
            print(f"warning: chunk overflow")
            if (
                    add_ellipsis_for_overflow
                    and len(tokenize(chunk_delimiter.join(candidate + ["..."]))) <= max_tokens
            ):
                candidate.append("...")
                dropped_chunk_count += 1
            continue  # this case would break downstream assumptions
        # estimate token count with the current chunk added
        extended_candidate_token_count = len(tokenize(chunk_delimiter.join(candidate + [chunk])))
        # If the token count exceeds max_tokens, add the current candidate to output and start a new candidate
        if extended_candidate_token_count > max_tokens:
            output.append(chunk_delimiter.join(candidate))
            output_indices.append(candidate_indices)
            candidate = chunk_with_header  # re-initialize candidate
            candidate_indices = [chunk_i]
        # otherwise keep extending the candidate
        else:
            candidate.append(chunk)
            candidate_indices.append(chunk_i)
    # add the remaining candidate to output if it's not empty
    if (header is not None and len(candidate) > 1) or (header is None and len(candidate) > 0):
        output.append(chunk_delimiter.join(candidate))
        output_indices.append(candidate_indices)
    return output, output_indices, dropped_chunk_count

def summarize(text: str,
              detail: float = 0,
              model: str = 'gpt-4-turbo',
              additional_instructions: Optional[str] = None,
              minimum_chunk_size: Optional[int] = 500,
              chunk_delimiter: str = ".",
              summarize_recursively=False,
              verbose=False):
    """
    Summarizes a given text by splitting it into chunks, each of which is summarized individually. 
    The level of detail in the summary can be adjusted, and the process can optionally be made recursive.

    Parameters:
    - text (str): The text to be summarized.
    - detail (float, optional): A value between 0 and 1 indicating the desired level of detail in the summary.
      0 leads to a higher level summary, and 1 results in a more detailed summary. Defaults to 0.
    - model (str, optional): The model to use for generating summaries. Defaults to 'gpt-3.5-turbo'.
    - additional_instructions (Optional[str], optional): Additional instructions to provide to the model for customizing summaries.
    - minimum_chunk_size (Optional[int], optional): The minimum size for text chunks. Defaults to 500.
    - chunk_delimiter (str, optional): The delimiter used to split the text into chunks. Defaults to ".".
    - summarize_recursively (bool, optional): If True, summaries are generated recursively, using previous summaries for context.
    - verbose (bool, optional): If True, prints detailed information about the chunking process.

    Returns:
    - str: The final compiled summary of the text.

    The function first determines the number of chunks by interpolating between a minimum and a maximum chunk count based on the `detail` parameter. 
    It then splits the text into chunks and summarizes each chunk. If `summarize_recursively` is True, each summary is based on the previous summaries, 
    adding more context to the summarization process. The function returns a compiled summary of all chunks.
    """

    # check detail is set correctly
    assert 0 <= detail <= 1

    # interpolate the number of chunks based to get specified level of detail
    max_chunks = len(chunk_on_delimiter(text, minimum_chunk_size, chunk_delimiter))
    min_chunks = 1
    num_chunks = int(min_chunks + detail * (max_chunks - min_chunks))

    # adjust chunk_size based on interpolated number of chunks
    document_length = len(tokenize(text))
    chunk_size = max(minimum_chunk_size, document_length // num_chunks)
    text_chunks = chunk_on_delimiter(text, chunk_size, chunk_delimiter)
    if verbose:
        print(f"Splitting the text into {len(text_chunks)} chunks to be summarized.")
        print(f"Chunk lengths are {[len(tokenize(x)) for x in text_chunks]}")

    # set system message
    system_message_content = "Rewrite this text in summarized form."
    if additional_instructions is not None:
        system_message_content += f"\n\n{additional_instructions}"

    accumulated_summaries = []
    for chunk in tqdm(text_chunks):
        if summarize_recursively and accumulated_summaries:
            # Creating a structured prompt for recursive summarization
            accumulated_summaries_string = '\n\n'.join(accumulated_summaries)
            user_message_content = f"Previous summaries:\n\n{accumulated_summaries_string}\n\nText to summarize next:\n\n{chunk}"
        else:
            # Directly passing the chunk for summarization without recursive context
            user_message_content = chunk

        # Constructing messages based on whether recursive summarization is applied
        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]

        # Assuming this function gets the completion and works as expected
        response = get_chat_completion(messages, model=model)
        accumulated_summaries.append(response)

    # Compile final summary from partial summaries
    final_summary = '\n\n'.join(accumulated_summaries)
    
    fixed_final = fix_summary(final_summary)

    return fixed_final

textA = ''' The post has just arrived and in it a very nice surprise, the discovery that Jacques Seguela, one-time adviser to President Mitterrand, now close confidant of President and Madame Sarkozy (indeed he intoduced them), and something of a legend in French political communications, has dedicated his latest book to little old moi. With apologies for the missing accents here and in the French bits of the long posting which follows - the dedication to 'Le Pouvoir dans la Peau' (Power in the skin) reads 'A Alastair Campbell, mon spin doctor prefere' (three missing accents in one word - mes excuses sinceres). So what did I do for this honour, you are asking? Well, perhaps the fact that he asked me to read his book, and write a 'postface' assessment both of his writing and of the issues he covers, and the fact that I said yes, has something to do with it. He says some blushmakingly kind things in his 'preface to the postface', which I will have to leave to French readers of the whole thing (published by Plon). But for the largely Anglophone visitors of this blog, I thought some of you might like to read the said 'postface' in English (apart from the bits where I quote direct from his book). I hope all those students who write asking for help with dissertations will find something quotable in it. Meanwhile I am off to Norway for a conference and a meeting with the Norwegian Labour Party. I'm looking forward to being in the country with the highest 'human development index' in the world, and which showed such a mature response to the recent massacre of Oslo and Utoya. Here is the postface to Le Pouvoir dans la Peau Jacques Seguela writes about political campaigns and communications not merely as an expert analyst, but as an experienced practitioner. Hence his latest book contains both insights worth heeding, but also enlivening tales of his own experience. He is observer and participant; outsider looking in, and insider looking out.  There is much to look at, not least in France with a Presidential election looming, and the outcome far from easy to predict. We live in a world defined by the pace of change, and whilst the velocity of that change has not always impacted upon our political institutions, many of which would remain recognisable to figures of history, it most certainly has impacted upon political communications. As Seguela writes: ‘En 5 ans le monde de la communication a plus evolue que dans les cents dernieres annees. ' Google, Youtube, Twitter, Facebook have quickly entered our language and changed the way we communicate, live our private lives, do business, do politics. People do not believe politicians as much as they once did. Nor do they believe the media. So who do we believe? We believe each other. The power and the political potential of social networks flows from that reality. Though fiercely modern in their application, social networks in some ways take us back to the politics of the village square. They are an electronic word of mouth on a sometimes global scale. This has changed the way people interact with each other and with their politicians. My first campaign as spokesman and strategist for Tony Blair was in 1997, three years in the planning after he had become leader of the Opposition  Labour Party. Some of the principles of strategy we applied back then would certainly apply to a modern day election. But their tactical execution almost certainly would not. Politicians and their strategists have to adapt to change as well as lead it. Seguela gives some interesting insights into those who have adapted well, and those who have done less well. He clearly adores former President Lula of Brazil and you can feel his yearning for a French leader who can somehow combine hard-headed strategy with human empathy in the same way as a man who left office with satisfaction ratings of 87percent. Seguela probably remains best known in political circles for his role advising Francois Mitterrand. Yet wheras I am 'tribal Labour', and could not imagine supporting a Conservative Party candidate in the UK, Seguela came out as a major supporter of Nicolas Sarkozy. I wonder if one of the reasons was not a frustration that large parts of the left in France remain eternally suspicious of modern communications techniques and styles which, frankly, no modern leader in a modern democracy can ignore. How he or she adapts to, or uses, them is up to them.'''

summarized =summarize(text=textA,
        detail=0.75, 
        model='gpt-4o-mini',
        minimum_chunk_size=500,
        summarize_recursively=True
    )

file = open('contents.txt', 'w')

file.write(summarized)

file.close()
