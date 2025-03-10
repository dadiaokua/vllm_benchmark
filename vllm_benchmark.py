import asyncio
import time

import numpy as np
from openai import AsyncOpenAI
import logging
import argparse
import json
import random

from util.util import some_endpoint_test

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

SHORT_PROMPTS = [
    "Explain the concept of artificial intelligence in simple terms.",
    "What are the main causes of climate change?",
    "Describe the process of photosynthesis in plants.",
    "How does the human immune system work?",
    "What were the main causes of World War II?",
    "Explain the theory of relativity in layman's terms.",
    "What are the key principles of effective leadership?",
    "How does blockchain technology work?",
    "What are the main theories about the origin of the universe?",
    "Describe the water cycle and its importance for life on Earth.",
    "What are the major differences between capitalism and socialism?",
    "How does the human brain process and store memories?",
    "What are the main challenges in space exploration?",
    "Explain the concept of supply and demand in economics.",
]

LONG_PROMPT_PAIRS = [
    {
        "prompt": "Explain the concept of artificial intelligence in simple terms.",
        "context": "Artificial intelligence (AI) is a rapidly evolving field of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence. These tasks include visual perception, speech recognition, decision-making, and language translation. AI systems are designed to learn from experience, adjust to new inputs, and perform human-like tasks. The field of AI encompasses various subfields, including machine learning, neural networks, and deep learning, which have led to significant advancements in areas such as autonomous vehicles, virtual assistants, and recommendation systems."
    },
    {
        "prompt": "What are the main causes of climate change?",
        "context": "Climate change is a complex global phenomenon primarily driven by human activities that release greenhouse gases into the atmosphere. The burning of fossil fuels for energy, deforestation, industrial processes, and agriculture are major contributors to the increased concentration of carbon dioxide and other heat-trapping gases. These gases form a 'blanket' around the Earth, causing the planet to warm at an unprecedented rate. The resulting changes in temperature patterns lead to more frequent and severe weather events, rising sea levels, and disruptions to ecosystems worldwide."
    },
    {
        "prompt": "Describe the process of photosynthesis in plants.",
        "context": "Photosynthesis is a fundamental biological process that allows plants to convert light energy into chemical energy. This process occurs in the chloroplasts of plant cells, specifically in structures called thylakoids. Chlorophyll, the pigment that gives plants their green color, is crucial in capturing light energy. During photosynthesis, plants take in carbon dioxide from the air through tiny pores called stomata and water from the soil through their roots. Using light energy, they combine these ingredients to produce glucose and oxygen. This process not only provides energy for the plant but also releases oxygen as a byproduct, which is essential for most life on Earth."
    },
    {
        "prompt": "How does the human immune system work?",
        "context": "The human immune system is a complex network of cells, tissues, and organs that work together to defend the body against harmful pathogens. It consists of two main parts: the innate immune system, which provides a quick, non-specific response to invaders, and the adaptive immune system, which develops targeted defenses against specific pathogens. Key components include white blood cells (such as neutrophils, macrophages, and lymphocytes), antibodies, and the complement system. The immune system has the remarkable ability to distinguish between the body's own cells and foreign invaders, allowing it to target threats while minimizing damage to healthy tissue."
    },
    {
        "prompt": "What were the main causes of World War II?",
        "context": "World War II, which lasted from 1939 to 1945, was one of the deadliest conflicts in human history. Its origins can be traced to several complex factors. The harsh terms of the Treaty of Versailles, which ended World War I, left Germany economically devastated and resentful. This paved the way for the rise of fascism and the Nazi Party under Adolf Hitler. Aggressive expansionist policies by Nazi Germany, Fascist Italy, and Imperial Japan, combined with the policy of appeasement by Western powers, allowed these regimes to gain territory unchecked. The immediate trigger for the war in Europe was Germany's invasion of Poland in September 1939, while the attack on Pearl Harbor in 1941 brought the United States into the conflict."
    },
    {
        "prompt": "Explain the theory of relativity in layman's terms.",
        "context": "Albert Einstein's theory of relativity, developed in the early 20th century, revolutionized our understanding of space, time, and gravity. It consists of two parts: special relativity and general relativity. Special relativity, introduced in 1905, deals with objects moving at very high speeds. It proposes that the speed of light is constant for all observers and that time and space are not absolute but relative to the observer's motion. This leads to phenomena like time dilation and length contraction. General relativity, published in 1915, extends these ideas to include gravity. Einstein proposed that massive objects curve the fabric of spacetime, and this curvature is what we experience as gravity. These theories have been consistently supported by experimental evidence and have practical applications in technologies like GPS satellites."
    },
    {
        "prompt": "What are the key principles of effective leadership?",
        "context": "Effective leadership is crucial in guiding organizations, teams, and individuals towards achieving their goals. While leadership styles may vary, several key principles are widely recognized as essential for success. These include clear communication, which ensures that vision and expectations are understood by all; integrity, which builds trust and respect; adaptability, allowing leaders to navigate changing environments; empathy, fostering strong relationships and understanding team dynamics; decision-making skills, enabling timely and informed choices; vision, providing direction and inspiration; and the ability to empower others, encouraging growth and innovation within the team. Effective leaders also demonstrate accountability, both for their own actions and those of their team, and continuously seek personal growth and learning opportunities."
    },
    {
        "prompt": "How does blockchain technology work?",
        "context": "Blockchain is a decentralized, distributed ledger technology that underlies cryptocurrencies like Bitcoin, but has potential applications far beyond digital currencies. At its core, a blockchain is a chain of blocks, each containing a list of transactions. Every block is linked to the previous one through cryptographic hashes, creating an immutable record. The key innovation of blockchain is its ability to achieve consensus in a decentralized network without requiring trust in any single entity. This is typically achieved through consensus mechanisms like Proof of Work or Proof of Stake. When a new transaction occurs, it is broadcast to a network of computers (nodes) for validation. Once validated, the transaction is combined with others to create a new block, which is then added to the chain. This process ensures transparency, security, and resistance to tampering, making blockchain suitable for various applications beyond finance, including supply chain management, voting systems, and digital identity verification."
    },
    {
        "prompt": "What are the main theories about the origin of the universe?",
        "context": "The origin of the universe has been a subject of intense scientific inquiry and philosophical debate for centuries. Currently, the most widely accepted scientific theory is the Big Bang model, which proposes that the universe began as an infinitely dense and hot singularity about 13.8 billion years ago, and has been expanding and cooling ever since. This theory is supported by observational evidence such as the cosmic microwave background radiation and the abundance of light elements in the universe. However, questions remain about what happened before the Big Bang and what caused it. Other theories include the Steady State theory, which suggests that the universe has always existed and is constantly creating new matter as it expands, though this theory has fallen out of favor due to lack of supporting evidence. More speculative ideas include the concept of a cyclic universe, where big bangs and big crunches occur in an endless cycle, and the idea of a multiverse, where our universe is just one of many existing universes."
    },
    {
        "prompt": "Describe the water cycle and its importance for life on Earth.",
        "context": "The water cycle, also known as the hydrologic cycle, is the continuous movement of water within the Earth and atmosphere. It is a complex system involving the processes of evaporation, transpiration, condensation, precipitation, and runoff. Water evaporates from the Earth's surface, primarily from oceans, lakes, and rivers, due to solar energy. Plants also release water vapor through transpiration. As this water vapor rises in the atmosphere, it cools and condenses to form clouds. Eventually, it falls back to Earth as precipitation in the form of rain, snow, or hail. Some of this water flows over the land as surface runoff, returning to bodies of water, while some seeps into the ground, replenishing groundwater reserves. This cycle is crucial for life on Earth as it redistributes water around the globe, shapes landscapes through erosion and deposition, regulates global temperatures, and provides fresh water essential for all living organisms. Understanding and protecting the water cycle is vital for managing water resources and addressing environmental challenges like climate change and water scarcity."
    },
    {
        "prompt": "What are the major differences between capitalism and socialism?",
        "context": "Capitalism and socialism are two contrasting economic and political systems that have shaped much of modern history. Capitalism is characterized by private ownership of the means of production, where individuals or corporations own businesses and property. It operates on the principles of free market competition, with prices determined by supply and demand. Profit is a key motivator in capitalist systems, and government intervention is generally limited. In contrast, socialism advocates for collective or governmental ownership and administration of the means of production and distribution of goods. It aims to create a more equitable society by reducing class distinctions and distributing resources according to need rather than ability to pay. In socialist systems, the government plays a much larger role in economic planning and the provision of social services. While pure forms of either system are rare, many countries adopt mixed economies incorporating elements of both capitalism and socialism to varying degrees."
    },
    {
        "prompt": "How does the human brain process and store memories?",
        "context": "The human brain's ability to process and store memories is a complex and fascinating process involving various regions and neural networks. When we experience something, sensory information is first processed in the relevant cortical areas (e.g., visual cortex for sight, auditory cortex for sound). This information is then integrated in the hippocampus, a seahorse-shaped structure crucial for forming new memories. The hippocampus helps bind different aspects of an experience into a cohesive memory and plays a key role in converting short-term memories into long-term ones. Long-term memories are thought to be stored through changes in synaptic connections between neurons across widespread areas of the cortex. This process, known as consolidation, can take days or even years. Different types of memories (e.g., episodic, semantic, procedural) involve different brain regions and processes. The retrieval of memories involves reactivating these neural patterns, which explains why memories can be influenced by our current state and environment. Understanding these processes is crucial for addressing memory-related disorders and developing potential therapies."
    },
    {
        "prompt": "What are the main challenges in space exploration?",
        "context": "Space exploration, while offering immense potential for scientific discovery and technological advancement, faces numerous challenges. One of the primary obstacles is the hostile environment of space itself. The vacuum of space, extreme temperatures, and harmful radiation pose significant risks to both human astronauts and sensitive equipment. Prolonged exposure to microgravity can lead to health issues for astronauts, including muscle atrophy and bone density loss. Logistical challenges are also substantial: the enormous distances involved in space travel require advanced propulsion systems and careful resource management. Launching payloads into orbit remains extremely expensive, limiting the scope and frequency of missions. Communication delays become increasingly problematic for deep space missions, necessitating a high degree of autonomy in spacecraft and rovers. Additionally, space debris orbiting Earth poses a growing threat to satellites and spacecraft. As we look towards long-term goals like establishing bases on the Moon or Mars, we face new challenges in creating sustainable habitats and managing psychological effects on crew members during extended missions. Despite these obstacles, ongoing research and technological innovations continue to push the boundaries of what's possible in space exploration."
    },
    {
        "prompt": "Explain the concept of supply and demand in economics.",
        "context": "Supply and demand is a fundamental concept in economics that describes how the price and quantity of a good or service in a market are determined through the interaction between buyers and sellers. The law of demand states that, all else being equal, as the price of a product increases, the quantity demanded by consumers decreases. This is typically represented by a downward-sloping demand curve. Conversely, the law of supply states that as the price of a product increases, the quantity that producers are willing to supply increases, represented by an upward-sloping supply curve. The point where these two curves intersect is called the equilibrium point, determining the market price and quantity. This model helps explain how prices fluctuate in response to changes in supply or demand. For instance, if demand increases while supply remains constant, prices will rise. If supply increases while demand remains constant, prices will fall. Understanding supply and demand is crucial for analyzing market behavior, predicting price changes, and formulating economic policies."
    },
    {
        "prompt": "What are the key features of a democratic government?",
        "context": "Democratic government is a system of governance based on the principle of rule by the people. While democracies can take various forms, they typically share several key features. First and foremost is the concept of free and fair elections, where citizens have the right to vote for their representatives at regular intervals. This is closely tied to the principle of political pluralism, allowing for multiple political parties and viewpoints to compete for power. The protection of individual rights and civil liberties, such as freedom of speech, press, and assembly, is another crucial aspect of democracy. Separation of powers is often implemented to prevent the concentration of power, typically dividing government into executive, legislative, and judicial branches that provide checks and balances on each other. The rule of law, ensuring that all citizens, including those in power, are equally subject to the law, is fundamental to democratic governance. Transparency and accountability in government operations, often facilitated by a free press and active civil society, help maintain democratic principles. Additionally, many democracies emphasize the protection of minority rights and the concept of majority rule with minority rights, aiming to balance the will of the majority with the fundamental rights of all citizens."
    },
    {
        "prompt": "How do vaccines work to prevent diseases?",
        "context": "Vaccines are one of the most effective tools in preventing infectious diseases, working by harnessing the body's own immune system. When a pathogen such as a virus or bacteria enters the body, the immune system responds by producing antibodies specific to that pathogen. These antibodies help neutralize or destroy the invader. Vaccines mimic this natural process by introducing a harmless form of the pathogen – either weakened, inactivated, or just a part of it – into the body. This stimulates the immune system to produce antibodies and memory cells specific to that pathogen, without causing the actual disease. If the vaccinated person later encounters the real pathogen, their immune system can quickly recognize it and mount a rapid and effective response, often preventing the disease entirely or reducing its severity. Some vaccines require multiple doses or periodic boosters to maintain immunity. The concept of herd immunity is also important in vaccination strategies: when a large portion of a population is vaccinated, it becomes difficult for the pathogen to spread, indirectly protecting those who cannot be vaccinated. Advances in vaccine technology, such as mRNA vaccines, are expanding our ability to rapidly develop vaccines for new threats."
    },
    {
        "prompt": "What are the main theories of human evolution?",
        "context": "Human evolution is the study of the biological and cultural development of our species, Homo sapiens, and our ancestors. The main scientific theory explaining human evolution is based on Darwin's theory of evolution by natural selection, adapted to incorporate modern genetic understanding. This theory proposes that humans evolved from earlier primate species over millions of years. Key ideas include the concept of common ancestry, suggesting that humans share a common ancestor with other primates, particularly the great apes. The 'Out of Africa' theory posits that modern humans originated in Africa and then migrated to other parts of the world. Fossil evidence has revealed a series of intermediate species, such as Australopithecus, Homo habilis, and Homo erectus, showing gradual changes in features like brain size, bipedalism, and tool use. Recent discoveries and genetic studies have complicated this picture, suggesting interbreeding between different human species (like Homo sapiens and Neanderthals) and the possibility of multiple migrations out of Africa. Ongoing research in paleontology, genetics, and archaeology continues to refine our understanding of human evolution, often challenging previous assumptions and revealing the complex history of our species."
    },
    {
        "prompt": "Describe the process of plate tectonics and its effects on Earth.",
        "context": "Plate tectonics is a fundamental theory in geology that explains the large-scale motions of Earth's lithosphere. The theory proposes that Earth's outer layer is divided into several large, rigid plates that move relative to one another. These plates float on the semi-fluid asthenosphere beneath them and are driven by convection currents in the mantle. Plate boundaries are classified into three types: divergent boundaries, where plates move apart and new crust is created; convergent boundaries, where plates collide, leading to subduction or mountain building; and transform boundaries, where plates slide past each other horizontally. The process of plate tectonics has profound effects on Earth's surface and internal structure. It is responsible for the formation of mountain ranges, ocean basins, and island arcs. It also plays a crucial role in the rock cycle, volcanic activity, and earthquake occurrence. Over geological time, plate tectonics has influenced climate patterns, ocean currents, and the distribution of flora and fauna across the globe. Understanding plate tectonics is essential for predicting geological hazards, explaining the distribution of natural resources, and comprehending Earth's long-term geological history."
    },
    {
        "prompt": "What are the primary causes of biodiversity loss?",
        "context": "Biodiversity loss, the decline in the variety of life forms on Earth, is a critical environmental issue with far-reaching consequences for ecosystems and human well-being. Several interconnected factors contribute to this loss. Habitat destruction and fragmentation, often due to human activities like deforestation, urbanization, and agricultural expansion, is a primary driver. Climate change is increasingly recognized as a major threat, altering ecosystems faster than many species can adapt. Overexploitation of natural resources, including overfishing and poaching, directly reduces populations of many species. Pollution, including chemical runoff, plastic waste, and air pollution, degrades habitats and harms wildlife. The introduction of invasive species, often facilitated by human activities, can disrupt local ecosystems and outcompete native species. Additionally, the spread of diseases, sometimes exacerbated by climate change and habitat stress, can devastate populations of certain species. These factors often interact and compound each other's effects, accelerating the rate of biodiversity loss. Addressing this crisis requires comprehensive conservation strategies, sustainable resource management, and global cooperation to mitigate human impacts on natural ecosystems."
    },
]


async def process_stream(stream):
    first_token_time = None
    total_tokens = 0
    async for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time()
        if chunk.choices[0].delta.content:
            total_tokens += 1
        if chunk.choices[0].finish_reason is not None:
            break
    return first_token_time, total_tokens


async def make_request(client, output_tokens, request_timeout, request, tokenizer):
    start_time = time.time()

    try:
        logging.getLogger("openai").setLevel(logging.ERROR)
        # 使用log_request=False参数来禁止在日志中打印请求内容
        stream = await client.chat.completions.create(
            model="llama_8b",
            messages=[
                {"role": "user", "content": request}
            ],
            max_tokens=output_tokens,
            stream=True
        )
        first_token_time, total_tokens = await asyncio.wait_for(process_stream(stream), timeout=request_timeout)
        end_time = time.time()
        elapsed_time = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else None
        input_token = tokenizer(request, truncation=False, return_tensors="pt").input_ids[0]
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
        logging.getLogger("openai").setLevel(logging.INFO)
        return total_tokens, elapsed_time, tokens_per_second, ttft, len(input_token)

    except asyncio.TimeoutError:
        logging.warning(f"Request timed out after {request_timeout} seconds")
        return None
    except Exception as e:
        logging.error(f"Error during request: {str(e)}")
        return None


async def worker(selected_clients, semaphore, queue, results, output_tokens, client_index, tokenizer,
                 request_timeout, rate_lambda, distribution, sample_content, config_round, worker_id):
    # 使用全局时间基准点，而不是相对于上一个请求的时间
    global_start_time = time.time()
    request_count = 0

    while True:
        task_id = await queue.get()
        if task_id is None:
            queue.task_done()
            break

        # 计算这个请求应该在什么时间点发送（基于全局开始时间）
        if distribution == "possion":
            # 泊松分布：请求之间的间隔遵循指数分布
            intervals = [float(np.random.exponential(1 / rate_lambda)) for _ in range(request_count + 1)]
            target_time = global_start_time + sum(intervals)
        elif distribution == "normal":
            # 正态分布：请求均匀分布，但有小的随机波动
            target_time = global_start_time + (request_count / rate_lambda) + float(np.random.normal(0, 0.01))
        else:
            # 均匀分布：请求基本均匀，但有一定范围的随机性
            jitter = np.random.uniform(-0.1, 0.1) / rate_lambda
            target_time = global_start_time + (request_count / rate_lambda) + jitter

        # 计算需要等待的时间
        now = time.time()
        wait_time = max(0, target_time - now)

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        # 记录实际请求发送时间
        actual_time = time.time()
        time_str = time.strftime('%H:%M:%S.%f', time.localtime(actual_time))[:-3]
        drift = actual_time - target_time

        logging.info(
            f"Worker {worker_id}, {config_round + 1} round task {task_id}: Actual={time_str}, "
            f"Target={time.strftime('%H:%M:%S.%f', time.localtime(target_time))[:-3]}, "
            f"Drift={drift:.3f}s, QPS target={rate_lambda}")

        # 随机选择一个请求内容
        request = random.choice(sample_content)

        # 根据task_id轮询选择客户端
        selected_client = selected_clients[task_id % len(selected_clients)]

        # 创建并发送异步请求，但不等待它完成
        await asyncio.create_task(
            process_request(selected_client, output_tokens, request_timeout, request, worker_id, tokenizer,
                            results, task_id, client_index, semaphore, config_round)
        )

        # 增加请求计数
        request_count += 1
        queue.task_done()


async def process_request(client, output_tokens, request_timeout, request, worker_id, tokenizer,
                          results, task_id, client_index, semaphore, config_round):
    async with semaphore:
        logging.info(f"Starting worker {worker_id} {config_round + 1} round request {task_id} for client {client_index}")
        try:
            result = await make_request(client, output_tokens, request_timeout, request, tokenizer)
            if result:
                results.append(result)
            else:
                logging.warning(f"Worker {worker_id} {config_round + 1} round Request {task_id} failed for client {client_index}")
        except Exception as e:
            logging.error(f"Worker {worker_id} {config_round + 1} round Request {task_id} for client {client_index} raised an exception: {e}")
        logging.info(f"Finished worker {worker_id} {config_round + 1} round request {task_id} for client {client_index}")


def calculate_percentile(values, percentile, reverse=False):
    if not values:
        return None
    if reverse:
        return np.percentile(values, 100 - percentile)
    return np.percentile(values, percentile)


async def run_benchmark(num_requests, concurrency, request_timeout,
                        output_tokens, clients, distribution, qps,
                        client_index, formatted_json, config_round, tokenizer):
    assert clients is not None, "Client must not be None"
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    # Calculate how many requests each worker should handle
    requests_per_worker = num_requests // concurrency
    remaining_requests = num_requests % concurrency

    # Create worker tasks - each worker gets a specific number of requests
    workers = []
    start_time = time.time()
    for worker_id in range(concurrency):
        # Distribute the remaining requests among the first few workers
        worker_requests = requests_per_worker + (1 if worker_id < remaining_requests else 0)

        # Calculate the task IDs this worker will handle
        start_id = worker_id * requests_per_worker + min(worker_id, remaining_requests)
        task_ids = list(range(start_id, start_id + worker_requests))

        # Create a dedicated queue for this worker
        worker_queue = asyncio.Queue()
        for task_id in task_ids:
            await worker_queue.put(task_id)
        await worker_queue.put(None)  # Sentinel to stop the worker

        # Create the worker task
        worker_task = asyncio.create_task(
            worker(clients, semaphore, worker_queue, results, output_tokens, client_index, tokenizer,
                   request_timeout, qps / concurrency, distribution, formatted_json, config_round, worker_id)
        )
        workers.append(worker_task)

    logging.info(f"Created {concurrency} workers, each handling approximately {requests_per_worker} requests")
    # Wait for all workers to complete
    await asyncio.gather(*workers)

    end_time = time.time()

    # Calculate metrics
    total_elapsed_time = end_time - start_time
    total_tokens = sum(tokens for tokens, _, _, _, _ in results if tokens is not None)
    total_input_tokens = sum(input_token for _, _, _, _, input_token in results if input_token is not None)
    latencies = [elapsed_time for _, elapsed_time, _, _, _ in results if elapsed_time is not None]
    tokens_per_second_list = [tps for _, _, tps, _, _ in results if tps is not None]
    ttft_list = [ttft for _, _, _, ttft, _ in results if ttft is not None]

    successful_requests = len(results)
    requests_per_second = successful_requests / total_elapsed_time if total_elapsed_time > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list) if tokens_per_second_list else 0
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0

    # Calculate percentiles
    percentiles = [50, 95, 99]
    latency_percentiles = [calculate_percentile(latencies, p) for p in percentiles]
    tps_percentiles = [calculate_percentile(tokens_per_second_list, p, reverse=True) for p in percentiles]
    ttft_percentiles = [calculate_percentile(ttft_list, p) for p in percentiles]

    return {
        "qps": qps,
        "total_requests": num_requests,
        "successful_requests": successful_requests,
        "concurrency": concurrency,
        "request_timeout": request_timeout,
        "max_output_tokens": output_tokens,
        "total_time": total_elapsed_time,
        "requests_per_second": requests_per_second,
        "total_output_tokens": total_tokens,
        "total_input_tokens": total_input_tokens,
        "latency": {
            "average": avg_latency,
            "p50": latency_percentiles[0],
            "p95": latency_percentiles[1],
            "p99": latency_percentiles[2]
        },
        "tokens_per_second": {
            "average": avg_tokens_per_second,
            "p50": tps_percentiles[0],
            "p95": tps_percentiles[1],
            "p99": tps_percentiles[2]
        },
        "time_to_first_token": {
            "average": avg_ttft,
            "p50": ttft_percentiles[0],
            "p95": ttft_percentiles[1],
            "p99": ttft_percentiles[2]
        },
        "client_index": client_index,
    }


def print_results(results):
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLaMA-3 model with vLLM")
    parser.add_argument("--num_requests", type=int, required=True, help="Number of requests to make", default=100)
    parser.add_argument("--concurrency", type=int, required=True, help="Number of concurrent requests", default=10)
    parser.add_argument("--request_timeout", type=int, default=20,
                        help="Timeout for each request in seconds (default: 30)")
    parser.add_argument("--output_tokens", type=int, default=50, help="Number of output tokens (default: 50)")
    parser.add_argument("--vllm_url", type=str, required=True, help="URL of the vLLM server",
                        default="http://222.201.144.119:8000")
    parser.add_argument("--api_key", type=str, required=True, help="API key for vLLM server", default='test')
    parser.add_argument("--use_long_context", action="store_true",
                        help="Use long context prompt pairs instead of short prompts", default=False)
    args = parser.parse_args()
    print(args)
    # server = None
    #
    # try:
    #     # 启动隧道
    #     server = start_tunnel()
    #
    #     # 测试 API 连接
    #     if some_endpoint_test(args.vllm_url):
    #         print("API connection successful. Proceeding with requests...")
    #     else:
    #         print("Failed to connect to API.")
    # finally:
    #     if server is not None:
    #         stop_tunnel(server)
    try:
        if some_endpoint_test(args.vllm_url):
            print("API connection successful. Proceeding with requests...")
        else:
            print("Failed to connect to API.")
    except Exception as e:
        print("Error to connect to API")
    client = AsyncOpenAI(base_url=args.vllm_url + "/v1")
    results = asyncio.run(
        run_benchmark(args.num_requests, args.concurrency, args.request_timeout, args.output_tokens, [client],
                      "passion", 1, 'short', 1, ""))
    if results == None:
        print("No results")
    else:
        print_results(results)
    # client = AsyncOpenAI(base_url=args.vllm_url)
    # print_results(results)
else:
    # When imported as a module, provide the run_benchmark function
    __all__ = ['run_benchmark']
