from openai import OpenAI
import uuid
import time
import threading
import requests
import asyncio
import concurrent.futures
from datetime import datetime
import os
import subprocess

# DP协调器KV缓存长度统计功能集成测试客户端
# DP Coordinator KV Cache Length Statistics Integration Test Client
#
# 测试目标：验证KV缓存长度统计信息能够从所有DP引擎正确收集到负载均衡器
# Test Goal: Verify that KV cache length statistics can be correctly collected
# from all DP engines to the load balancer

# 全局变量用于监控统计
monitoring_active = True
stats_history = []
log_monitoring_active = True

def monitor_kv_cache_stats():
    """监控KV缓存统计信息的后台线程函数"""
    print("🔍 开始监控KV缓存统计信息...")

    while monitoring_active:
        time.sleep(5)  # 每5秒监控一次

def monitor_log_files():
    """实时监控日志文件中的KV缓存统计信息"""
    print("📋 开始实时监控日志文件中的KV缓存统计信息...")

    log_files = ["decode_dp.log", "prefill_dp.log"]
    last_positions = {log_file: 0 for log_file in log_files}

    while log_monitoring_active:
        for log_file in log_files:
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        f.seek(last_positions[log_file])
                        new_lines = f.readlines()
                        last_positions[log_file] = f.tell()

                        for line in new_lines:
                            # 只显示DPLBAsyncMPClient的关键监控信息
                            if any(keyword in line for keyword in [
                                '🎯 DPLBAsyncMPClient监控更新', '⚠️ DPLBAsyncMPClient监控更新'
                            ]):
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                print(f"📋 [{timestamp}] {line.strip()}")

                except Exception as e:
                    # 忽略文件读取错误，继续监控
                    pass

        time.sleep(1)  # 每秒检查一次日志文件

def create_dynamic_prompt(base_prompt, target_tokens):
    """根据目标token数动态生成prompt"""
    # 估算当前prompt的token数（中文约3字符/token）
    current_tokens = len(base_prompt) // 3

    if current_tokens >= target_tokens:
        return base_prompt

    # 需要扩展的token数
    tokens_needed = target_tokens - current_tokens
    chars_needed = tokens_needed * 3

    # 扩展内容模板 - 更长的内容
    extensions = [
        "请在回答中包含具体的技术细节和实现方法，详细说明每个步骤的操作流程和关键参数设置。",
        "同时提供相关的代码示例和最佳实践，包括完整的实现代码、配置文件示例和运行结果分析。",
        "分析该技术在不同应用场景下的性能表现和优化策略，包括计算复杂度分析、内存使用优化、并发处理能力等方面的详细评估。",
        "讨论该领域的最新研究进展和未来发展趋势，引用近期的重要学术论文和工业界的创新实践案例。",
        "对比不同技术方案的优缺点和适用条件，从性能、可扩展性、维护成本、学习曲线等多个维度进行全面比较。",
        "说明在实际项目中的部署经验和注意事项，包括环境配置、依赖管理、性能调优、故障排除等实践经验。",
        "提供详细的数学公式推导和理论分析，包括算法的时间复杂度和空间复杂度分析，以及收敛性证明。",
        "举例说明在工业界的成功应用案例，包括具体的业务场景、技术架构、实施过程和取得的效果。",
        "分析可能遇到的技术挑战和解决方案，包括常见的错误类型、调试方法、性能瓶颈识别和优化策略。",
        "讨论该技术对相关领域的影响和意义，分析其在推动行业发展和技术创新方面的重要作用。",
        "请结合最新的学术论文和研究成果进行分析，引用权威的数据和实验结果来支持你的观点。",
        "说明该技术的发展历程和关键里程碑，梳理从早期概念到现在成熟应用的演进过程。",
        "分析不同参数设置对结果的影响，提供参数调优的指导原则和经验总结。",
        "讨论该技术在不同数据规模下的表现，包括小规模、中等规模和大规模数据集的处理能力对比。",
        "提供性能评估指标和基准测试结果，包括准确率、召回率、F1分数、处理速度等关键指标的详细分析。",
        "说明与其他相关技术的集成方法，包括API接口设计、数据格式转换、系统架构整合等技术细节。",
        "分析计算复杂度和资源消耗情况，包括CPU使用率、内存占用、网络带宽需求等资源消耗的详细分析。",
        "讨论可扩展性和并行化实现方案，包括水平扩展、垂直扩展、分布式部署等不同扩展策略的比较。",
        "提供故障排除和调试技巧，包括日志分析、性能监控、错误诊断等实用的运维经验。",
        "说明标准化和规范化的重要性，包括代码规范、文档标准、测试规范等软件工程最佳实践。"
    ]

    # 更详细的扩展内容
    detailed_extensions = [
        "请确保回答内容结构清晰，逻辑严密，包含引言、主体和结论部分。在引言部分简要介绍背景和目标，在主体部分详细阐述核心内容，在结论部分总结要点和展望未来。",
        "在技术解释中使用准确的专业术语，并提供必要的背景知识。对于复杂的概念，请先给出定义，然后通过类比和示例来帮助理解。",
        "通过具体的数据和实验结果来支持你的观点和结论。引用可靠的数据源，提供详细的实验设计和结果分析。",
        "考虑不同读者的技术背景，提供适当的解释深度。对于初学者，提供更多的基础知识；对于专家，重点关注高级特性和最新发展。",
        "引用权威的学术资源和行业标准来增强内容的可信度。包括顶级会议论文、知名期刊文章、官方文档等可靠来源。",
        "讨论该技术的局限性和改进空间，保持客观的分析态度。不仅要说明优点，也要诚实地指出存在的问题和挑战。",
        "提供实用的建议和指导，帮助读者在实际工作中应用相关知识。包括具体的操作步骤、配置参数、注意事项等。",
        "使用图表、流程图或示例代码来辅助说明复杂的概念。通过可视化的方式让抽象的概念更容易理解。",
        "分析该技术对行业发展和社会进步的潜在影响。从经济效益、社会价值、环境影响等多个角度进行综合评估。",
        "讨论相关的伦理问题和社会责任考虑。特别是在AI和数据处理领域，要关注隐私保护、算法公平性等重要议题。"
    ]

    # 开始扩展prompt
    extended_prompt = base_prompt
    import random

    # 首先添加基础扩展
    random.shuffle(extensions)
    for ext in extensions:
        if len(extended_prompt) >= target_tokens * 3:
            break
        extended_prompt += " " + ext

    # 如果还不够长，添加详细扩展
    if len(extended_prompt) < target_tokens * 3:
        random.shuffle(detailed_extensions)
        for ext in detailed_extensions:
            if len(extended_prompt) >= target_tokens * 3:
                break
            extended_prompt += " " + ext

    # 如果还是不够长，重复添加内容
    while len(extended_prompt) < target_tokens * 3:
        additional_content = "请提供更多的技术细节、实现方案、应用案例和最佳实践经验。"
        extended_prompt += " " + additional_content
        if len(additional_content) * 10 > chars_needed:  # 避免无限循环
            break

    return extended_prompt

def create_long_tail_prompts():
    """创建符合长尾分布的真实测试prompt集合"""
    import random

    # 基础prompt模板
    base_prompts = [
        "什么是人工智能？请详细解释AI的定义、发展历程和主要应用领域。",
        "解释机器学习的基本原理和主要算法类型。",
        "深度学习相比传统机器学习有哪些优势？",
        "自然语言处理技术的核心组件和应用场景有哪些？",
        "请介绍Transformer模型的核心创新点和技术优势。",
        "什么是注意力机制？请全面解释其概念和应用。",
        "分布式训练在大规模机器学习中的重要性和实现方法。",
        "大语言模型的主要特点和技术架构。",
        "神经网络的工作原理和基本组成结构。",
        "梯度下降算法的数学原理和优化变种。",
        "卷积神经网络在计算机视觉中的应用。",
        "循环神经网络处理序列数据的机制。",
        "强化学习的基本概念和算法框架。",
        "生成对抗网络的工作原理和应用场景。",
        "迁移学习在深度学习中的重要作用。"
    ]

    # 根据不同长度要求生成prompt
    short_prompts = [create_dynamic_prompt(prompt, random.randint(100, 300)) for prompt in base_prompts]
    medium_prompts = [create_dynamic_prompt(prompt, random.randint(500, 1000)) for prompt in base_prompts]
    long_prompts = [create_dynamic_prompt(prompt, random.randint(1000, 3000)) for prompt in base_prompts]

    return short_prompts, medium_prompts, long_prompts

def generate_long_tail_requests(total_requests=100, duration_seconds=10, random_seed=42):
    """
    生成符合长尾分布的真实测试请求序列

    Args:
        total_requests: 总请求数
        duration_seconds: 请求分散时间
        random_seed: 随机种子，确保可复现

    Returns:
        List of request dictionaries with proper distribution:
        - 80% SHORT requests (100-300 tokens input)
        - 15% MEDIUM requests (500-1000 tokens input)
        - 5% LONG requests (1000-3000 tokens input)
    """
    import random

    # 设置随机种子确保可复现
    random.seed(random_seed)
    print(f"🎲 设置随机种子: {random_seed}")

    # 生成不同长度的prompt模板
    print("📝 生成prompt模板...")
    short_prompts, medium_prompts, long_prompts = create_long_tail_prompts()

    print(f"✅ Prompt模板生成完成:")
    print(f"   - 短prompt模板: {len(short_prompts)}个")
    print(f"   - 中等prompt模板: {len(medium_prompts)}个")
    print(f"   - 长prompt模板: {len(long_prompts)}个")

    # 预先计算每种类型的请求数量
    expected_short_count = int(total_requests * 0.8)  # 80%
    expected_medium_count = int(total_requests * 0.15)  # 15%
    expected_long_count = total_requests - expected_short_count - expected_medium_count  # 剩余的5%

    print(f"📊 预期请求分布:")
    print(f"   - 短请求: {expected_short_count}个 ({expected_short_count/total_requests*100:.1f}%)")
    print(f"   - 中等请求: {expected_medium_count}个 ({expected_medium_count/total_requests*100:.1f}%)")
    print(f"   - 长请求: {expected_long_count}个 ({expected_long_count/total_requests*100:.1f}%)")

    # 创建请求类型列表，确保精确的分布
    request_types = (
        ['SHORT'] * expected_short_count +
        ['MEDIUM'] * expected_medium_count +
        ['LONG'] * expected_long_count
    )

    # 打乱请求类型顺序
    random.shuffle(request_types)

    print(f"🔀 请求类型序列已打乱，开始生成具体请求...")

    requests = []
    type_counters = {'SHORT': 0, 'MEDIUM': 0, 'LONG': 0}

    for i in range(total_requests):
        request_type = request_types[i]
        type_counters[request_type] += 1

        # 根据请求类型选择对应的prompt和参数
        if request_type == "SHORT":
            # 短请求: 100-300 tokens输入，100-300 tokens输出
            prompt = random.choice(short_prompts)
            max_tokens = random.randint(100, 300)
            expected_input_range = "100-300"
        elif request_type == "MEDIUM":
            # 中等请求: 500-1000 tokens输入，300-500 tokens输出
            prompt = random.choice(medium_prompts)
            max_tokens = random.randint(300, 500)
            expected_input_range = "500-1000"
        else:  # LONG
            # 长请求: 1000-3000 tokens输入，500-1000 tokens输出
            prompt = random.choice(long_prompts)
            max_tokens = random.randint(500, 1000)
            expected_input_range = "1000-3000"

        # 计算实际prompt长度（中文约3字符/token）
        actual_prompt_tokens = len(prompt) // 3

        # 生成唯一的请求ID
        request_id = f"longtail_{request_type.lower()}_{type_counters[request_type]:03d}_{uuid.uuid4().hex[:6]}"

        # 创建请求对象
        request_obj = {
            'prompt': prompt,
            'max_tokens': max_tokens,
            'request_id': request_id,
            'request_type': request_type,
            'estimated_prompt_tokens': actual_prompt_tokens,
            'expected_input_range': expected_input_range
        }

        requests.append(request_obj)

        # 每10个请求显示一次进度
        if (i + 1) % 10 == 0 or (i + 1) == total_requests:
            print(f"📈 生成进度: {i + 1}/{total_requests} ({(i + 1)/total_requests*100:.1f}%)")

    # 验证最终分布
    final_short_count = sum(1 for req in requests if req['request_type'] == 'SHORT')
    final_medium_count = sum(1 for req in requests if req['request_type'] == 'MEDIUM')
    final_long_count = sum(1 for req in requests if req['request_type'] == 'LONG')

    print(f"✅ 请求生成完成，最终分布验证:")
    print(f"   - 短请求: {final_short_count}个 ({final_short_count/total_requests*100:.1f}%)")
    print(f"   - 中等请求: {final_medium_count}个 ({final_medium_count/total_requests*100:.1f}%)")
    print(f"   - 长请求: {final_long_count}个 ({final_long_count/total_requests*100:.1f}%)")

    # 验证token长度分布
    short_tokens = [req['estimated_prompt_tokens'] for req in requests if req['request_type'] == 'SHORT']
    medium_tokens = [req['estimated_prompt_tokens'] for req in requests if req['request_type'] == 'MEDIUM']
    long_tokens = [req['estimated_prompt_tokens'] for req in requests if req['request_type'] == 'LONG']

    if short_tokens:
        print(f"📏 短请求token长度: {min(short_tokens)}-{max(short_tokens)} (目标: 100-300)")
    if medium_tokens:
        print(f"📏 中等请求token长度: {min(medium_tokens)}-{max(medium_tokens)} (目标: 500-1000)")
    if long_tokens:
        print(f"📏 长请求token长度: {min(long_tokens)}-{max(long_tokens)} (目标: 1000-3000)")

    return requests

def process_single_request(client, model, prompt, request_id, max_tokens):
    """处理单个请求并记录结果"""
    start_time = time.time()
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    try:
        # 估算prompt的token数量（粗略估算：中文约2-3字符/token，英文约4-5字符/token）
        estimated_prompt_tokens = len(prompt) // 3  # 粗略估算
        print(f"🚀 [{timestamp}] 请求 {request_id}: 开始处理")
        print(f"   📝 Prompt长度: {len(prompt)}字符 (估算~{estimated_prompt_tokens}tokens)")
        print(f"   🎯 目标生成: {max_tokens}tokens")
        print(f"   📊 预期KV缓存长度变化: {estimated_prompt_tokens} → {estimated_prompt_tokens + max_tokens}")

        # 发送请求到decode实例
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            extra_headers={"X-Request-Id": request_id}
        )

        end_time = time.time()
        duration = end_time - start_time
        timestamp_end = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        generated_text = response.choices[0].text
        generated_tokens = len(generated_text.split())  # 粗略估算token数
        estimated_prompt_tokens = len(prompt) // 3
        total_kv_cache_tokens = estimated_prompt_tokens + generated_tokens

        print(f"✅ [{timestamp_end}] 请求 {request_id}: 完成处理")
        print(f"   ⏱️  处理耗时: {duration:.2f}秒")
        print(f"   📊 Token统计: prompt~{estimated_prompt_tokens}, 生成~{generated_tokens}, 总计~{total_kv_cache_tokens}")
        print(f"   💾 预期KV缓存长度: {total_kv_cache_tokens} tokens")
        print(f"   📝 生成内容: {generated_text[:80]}{'...' if len(generated_text) > 80 else ''}")

        return {
            'request_id': request_id,
            'success': True,
            'duration': duration,
            'prompt_length': len(prompt),
            'generated_length': len(generated_text),
            'estimated_tokens': generated_tokens,
            'estimated_prompt_tokens': estimated_prompt_tokens,
            'total_kv_cache_tokens': total_kv_cache_tokens
        }

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        timestamp_error = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        print(f"❌ [{timestamp_error}] 请求 {request_id}: 处理失败 - {str(e)}")
        return {
            'request_id': request_id,
            'success': False,
            'duration': duration,
            'error': str(e)
        }

try:
    print("🔍 开始KV缓存感知调度算法测试...")

    # 1: 设置decode实例的客户端（主要测试decode阶段的KV缓存统计）
    openai_api_key = "EMPTY"  # vLLM不需要真实的API密钥

    decode_client = OpenAI(
        api_key=openai_api_key,
        base_url="http://127.0.0.1:8200/v1",  # Decode实例URL - 直接测试decode实例
        timeout=120.0  # 增加超时时间以处理并发请求
    )
    
    # 获取decode实例的模型名称
    models = decode_client.models.list()
    model = models.data[0].id
    print(f"🤖 使用模型: {model}")

    # 2: KV缓存感知调度算法真实负载测试
    print("\n" + "="*80)
    print("🧪 开始KV缓存感知调度算法真实负载测试")
    print("🎯 测试目标: 验证KV缓存感知调度算法在真实负载下的性能表现")
    print("📊 观察重点: 调度决策过程、引擎负载分布、系统吞吐量")
    print("🔥 负载特征: 100个请求，10 QPS，长尾分布，真实prompt长度")
    print("="*80)

    # 生成长尾分布的测试请求（使用固定随机种子确保可复现）
    total_requests = 100  # 总请求数
    duration_seconds = 10  # 请求分散在10秒内到达，平均QPS=10
    random_seed = 42  # 固定随机种子

    print(f"🎯 开始生成长尾分布测试请求...")
    requests_to_process = generate_long_tail_requests(total_requests, duration_seconds, random_seed)

    print(f"📋 真实负载测试配置:")
    print(f"   - 总请求数: {total_requests}")
    print(f"   - 请求到达时间: {duration_seconds}秒内分散到达")
    print(f"   - 平均QPS: {total_requests/duration_seconds:.1f} (每秒{total_requests/duration_seconds:.0f}个请求)")
    print(f"   - 输入长度: 80%短请求(100-300tokens), 15%中等请求(500-1000tokens), 5%长请求(1000-3000tokens)")
    print(f"   - 输出长度: 短请求100-300tokens, 中等请求300-500tokens, 长请求500-1000tokens")
    print(f"   - 随机种子: 42 (确保可复现)")

    # 统计请求分布和预期负载
    short_count = sum(1 for req in requests_to_process if req['request_type'] == 'SHORT')
    medium_count = sum(1 for req in requests_to_process if req['request_type'] == 'MEDIUM')
    long_count = sum(1 for req in requests_to_process if req['request_type'] == 'LONG')

    print(f"📊 实际生成的请求分布:")
    print(f"   - 短请求: {short_count}个 ({short_count/total_requests*100:.1f}%) - 输入100-300tokens, 输出100-300tokens")
    print(f"   - 中等请求: {medium_count}个 ({medium_count/total_requests*100:.1f}%) - 输入500-1000tokens, 输出300-500tokens")
    print(f"   - 长请求: {long_count}个 ({long_count/total_requests*100:.1f}%) - 输入1000-3000tokens, 输出500-1000tokens")

    print(f"\n📊 准备发送 {len(requests_to_process)} 个真实负载请求...")

    # 使用新的token估算方式
    total_estimated_prompt_tokens = sum(req['estimated_prompt_tokens'] for req in requests_to_process)
    total_estimated_generated_tokens = sum(req['max_tokens'] for req in requests_to_process)
    total_estimated_kv_tokens = total_estimated_prompt_tokens + total_estimated_generated_tokens

    print(f"\n📈 预期负载统计:")
    print(f"   - 总输入tokens: ~{total_estimated_prompt_tokens:,}")
    print(f"   - 总输出tokens: ~{total_estimated_generated_tokens:,}")
    print(f"   - 总KV缓存tokens: ~{total_estimated_kv_tokens:,}")
    print(f"   - 平均每请求KV缓存: ~{total_estimated_kv_tokens//total_requests:,} tokens")

    # 3: 执行时间分散的请求测试
    print(f"\n🚀 开始执行时间分散的长尾分布请求...")
    start_time = datetime.now()

    # 使用线程池执行请求，但按时间分散发送
    max_workers = min(20, len(requests_to_process))  # 限制最大并发数
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_request = {}

        # 请求已经按生成顺序排列，无需额外排序

        print(f"📤 将在{duration_seconds}秒内分散发送{len(requests_to_process)}个请求 (平均QPS: {len(requests_to_process)/duration_seconds:.1f})...")
        print("⏳ 真实负载测试进行中，调度决策日志将记录在decode_dp.log中...")
        print("🔍 关键信息: 每个请求的调度决策原因、选择的引擎、系统状态")

        # 分散发送请求
        request_start_time = time.time()  # 用于计算相对时间
        for i, req_info in enumerate(requests_to_process):
            # 计算相对到达时间
            relative_arrival_time = (i / len(requests_to_process)) * duration_seconds
            elapsed_time = time.time() - request_start_time
            wait_time = relative_arrival_time - elapsed_time
            if wait_time > 0:
                time.sleep(wait_time)

            # 提交请求
            future = executor.submit(
                process_single_request,
                decode_client,
                model,
                req_info['prompt'],
                req_info['request_id'],
                req_info['max_tokens']
            )
            future_to_request[future] = req_info

            # 简化请求提交信息
            if len(future_to_request) % 10 == 0:  # 每10个请求显示一次进度
                print(f"📨 已提交 {len(future_to_request)}/{len(requests_to_process)} 个请求...")

        # 收集结果
        results = []
        completed_count = 0

        for future in concurrent.futures.as_completed(future_to_request):
            req_info = future_to_request[future]
            result = future.result()
            results.append(result)
            completed_count += 1

            # 每10个请求显示一次进度
            if completed_count % 10 == 0 or completed_count == len(requests_to_process):
                progress = (completed_count / len(requests_to_process)) * 100
                print(f"📈 进度: {completed_count}/{len(requests_to_process)} ({progress:.1f}%) 完成")

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    print(f"\n🎉 所有并发请求处理完成!")
    print(f"⏱️  总耗时: {total_duration:.2f}秒")
    
    # 4: 分析测试结果
    print(f"\n📊 测试结果分析:")
    print("="*60)

    successful_requests = [r for r in results if r['success']]
    failed_requests = [r for r in results if not r['success']]

    print(f"✅ 成功请求: {len(successful_requests)}/{len(results)}")
    print(f"❌ 失败请求: {len(failed_requests)}/{len(results)}")

    if successful_requests:
        avg_duration = sum(r['duration'] for r in successful_requests) / len(successful_requests)
        total_generated_tokens = sum(r.get('estimated_tokens', 0) for r in successful_requests)
        total_prompt_tokens = sum(r.get('estimated_prompt_tokens', 0) for r in successful_requests)
        total_kv_cache_tokens = sum(r.get('total_kv_cache_tokens', 0) for r in successful_requests)

        print(f"⏱️  平均处理时间: {avg_duration:.2f}秒")
        print(f"📊 Token统计汇总:")
        print(f"   - 总Prompt tokens: ~{total_prompt_tokens}")
        print(f"   - 总生成tokens: ~{total_generated_tokens}")
        print(f"   - 总KV缓存tokens: ~{total_kv_cache_tokens}")

        print(f"\n📈 详细结果:")
        for result in successful_requests:
            print(f"   🔸 {result['request_id']}: "
                  f"{result['duration']:.2f}s, "
                  f"prompt:{result['prompt_length']}字符(~{result.get('estimated_prompt_tokens', 0)}tokens), "
                  f"生成:~{result.get('estimated_tokens', 0)}tokens, "
                  f"KV缓存:~{result.get('total_kv_cache_tokens', 0)}tokens")

    if failed_requests:
        print(f"\n❌ 失败请求详情:")
        for result in failed_requests:
            print(f"   🔸 {result['request_id']}: {result.get('error', 'Unknown error')}")

    # 5: 继续监控DP协调器KV缓存统计信息
    print(f"\n🔍 继续监控DP协调器KV缓存统计信息30秒...")
    print("📋 重点观察:")
    print("   - decode_dp.log中的'📊 收到DP统计更新'条目")
    print("   - '💾 KV缓存长度 (本客户端管理的引擎)'的数值变化")
    print("   - '📈 全局总KV缓存长度'的增长和回落过程")
    print("   - waiting/running队列的动态变化")
    print("   - 验证lb_engines_tokens是否正确收集了统计数据")

    # 保持监控30秒
    for i in range(10):
        time.sleep(3)
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"📊 [{timestamp}] DP协调器统计监控中... ({i+1}/10)")

    # 停止监控线程
    monitoring_active = False
    log_monitoring_active = False

    print(f"\n🎉 DP协调器KV缓存长度统计功能集成测试完成！")
    print("📊 请检查日志文件中的DP协调器KV缓存统计信息:")
    print("   - decode_dp.log: Decode引擎的DP协调器统计")
    print("   - 查找包含'收到DP统计更新'、'KV缓存长度'、'lb_engines_tokens'的日志条目")
    print("   - 验证KV缓存长度统计信息是否从DP引擎正确流向负载均衡器")
    print("   - 观察KV缓存长度从初始值增长到峰值再回落的完整过程")

    # 等待日志监控线程结束
    time.sleep(2)

except Exception as e:
    print(f"❌ DP协调器KV缓存长度统计功能集成测试过程中出错: {e}")
    print("请检查decode实例是否正在运行且可访问")
    print("确保PD分离架构已正确部署并且服务正常运行")
    import traceback
    traceback.print_exc()

    # 停止监控线程
    monitoring_active = False
    log_monitoring_active = False

