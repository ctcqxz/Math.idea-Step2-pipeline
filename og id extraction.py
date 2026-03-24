"""
OG ID Extraction Pipeline
从论文LaTeX代码中提取核心证明步骤的结构化表示
"""

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class PaperOverview:
    """论文概览信息"""
    research_field: str = ""
    research_problem: str = ""
    main_results: str = ""
    proof_outline: str = ""  # 不超过300字


@dataclass
class GlobalAssumptions:
    """全局假设"""
    assumptions: List[str] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class Theorem:
    """定理及其证明"""
    theorem_id: str = ""
    statement: str = ""
    proof: str = ""
    latex_source: str = ""


@dataclass
class KeyStep:
    """关键困难步骤"""
    step_id: str = ""
    theorem_id: str = ""
    description: str = ""
    proof_segment: str = ""
    importance_score: int = 0


@dataclass
class Idea:
    """核心技巧/思路"""
    source: str = ""  # "technique_analysis" or "human_thinking"
    description: str = ""
    details: str = ""


@dataclass
class OgId:
    """最终输出的 OG ID 结构"""
    og_id: str = ""
    global_assumptions: List[str] = field(default_factory=list)
    mathematical_objects: List[str] = field(default_factory=list)
    conditional_assumptions: List[str] = field(default_factory=list)
    known_conclusions: List[str] = field(default_factory=list)
    proof_goal: str = ""
    source_theorem_id: str = ""
    source_step_id: str = ""
    conciseness_score: int = 0
    accuracy_score: int = 0


# ============================================================
# LLM 调用封装
# ============================================================

class LLMClient:
    """大模型 API 客户端封装"""

    def __init__(self, api_key: str, model: str = "gpt-4", base_url: str = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        # 实际项目中这里初始化 openai client 或其他 SDK
        # import openai
        # self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def call(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        """
        调用大模型 API
        返回模型的文本响应
        """
        # ---- 实际调用示例 (以 OpenAI 兼容接口为例) ----
        # response = self.client.chat.completions.create(
        #     model=self.model,
        #     temperature=temperature,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt},
        #     ]
        # )
        # return response.choices[0].message.content

        # 占位实现
        raise NotImplementedError("请接入实际的大模型 API")

    def call_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        """
        调用大模型 API 并解析 JSON 返回
        """
        raw = self.call(system_prompt, user_prompt, temperature)
        # 尝试从返回中提取 JSON
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(text: str) -> Any:
        """从模型返回文本中提取 JSON"""
        # 尝试直接解析
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 块
        import re
        pattern = r'```(?:json)?\s*\n?(.*?)\n?\s*```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试找到第一个 [ 或 { 开始的部分
        for i, ch in enumerate(text):
            if ch in ('{', '['):
                try:
                    return json.loads(text[i:])
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"无法从模型输出中解析 JSON:\n{text[:500]}")


# ============================================================
# 核心流水线
# ============================================================

class OgIdPipeline:
    """OG ID 提取流水线"""

    MAX_LOOP = 10  # 最大循环次数

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_theorems": 0,
            "total_steps": 0,
            "total_og_ids": 0,
            "discarded_steps": 0,
            "discarded_og_ids": 0,
        }

    # ----------------------------------------------------------------
    # Step 2: 解析论文概览
    # ----------------------------------------------------------------
    def step2_parse_overview(self, latex_code: str) -> PaperOverview:
        """解析论文的研究领域、研究问题、主要结果、证明路线"""
        logger.info("Step 2: 解析论文概览...")

        system_prompt = "提示词2_system"
        # 提示词2: 你是一位数学论文分析专家。请阅读以下LaTeX源代码，提取并输出：
        # ① research_field: 研究领域
        # ② research_problem: 研究问题
        # ③ main_results: 主要结果
        # ④ proof_outline: 证明路线（简述，不超过300字）
        # 以 JSON 格式输出。

        user_prompt = f"提示词2_user\n\n以下是论文的LaTeX代码：\n\n{latex_code}"

        result = self.llm.call_json(system_prompt, user_prompt)

        overview = PaperOverview(
            research_field=result.get("research_field", ""),
            research_problem=result.get("research_problem", ""),
            main_results=result.get("main_results", ""),
            proof_outline=result.get("proof_outline", ""),
        )

        logger.info(f"  研究领域: {overview.research_field}")
        logger.info(f"  研究问题: {overview.research_problem[:80]}...")
        return overview

    # ----------------------------------------------------------------
    # Step 3: 提取全局假设
    # ----------------------------------------------------------------
    def step3_extract_global_assumptions(self, latex_code: str) -> GlobalAssumptions:
        """提取论文中的全局假设"""
        logger.info("Step 3: 提取全局假设...")

        system_prompt = "提示词3_system"
        # 提示词3: 你是一位数学论文分析专家。请阅读以下LaTeX源代码，提取论文中的全局假设。
        # 全局假设是指贯穿全文的基本设定，例如：环是否有单位元、是否可交换、
        # 域的特征、拓扑空间的分离性假设等。
        # 以 JSON 格式输出：{"assumptions": ["假设1", "假设2", ...], "raw_text": "原文相关段落"}

        user_prompt = f"提示词3_user\n\n以下是论文的LaTeX代码：\n\n{latex_code}"

        result = self.llm.call_json(system_prompt, user_prompt)

        assumptions = GlobalAssumptions(
            assumptions=result.get("assumptions", []),
            raw_text=result.get("raw_text", ""),
        )

        logger.info(f"  全局假设数量: {len(assumptions.assumptions)}")
        for a in assumptions.assumptions:
            logger.info(f"    - {a}")
        return assumptions

    # ----------------------------------------------------------------
    # Step 4: 切分并提取含有核心步骤的定理
    # ----------------------------------------------------------------
    def step4_extract_theorems(self, latex_code: str) -> List[Theorem]:
        """切分并提取论文中所有含有核心步骤的定理"""
        logger.info("Step 4: 切分并提取定理...")

        system_prompt = "提示词4_system"
        # 提示词4: 你是一位数学论文分析专家。请阅读以下LaTeX源代码，
        # 提取论文中所有含有核心步骤的定理（核心步骤是指具有本质困难的步骤）。
        # 对每个定理，提取其完整的定理陈述和证明。
        # 以 JSON 数组格式输出：
        # [{"theorem_id": "Theorem 1.1", "statement": "...", "proof": "...", "latex_source": "..."}, ...]

        user_prompt = f"提示词4_user\n\n以下是论文的LaTeX代码：\n\n{latex_code}"

        result = self.llm.call_json(system_prompt, user_prompt)

        theorems = []
        for item in result:
            thm = Theorem(
                theorem_id=item.get("theorem_id", ""),
                statement=item.get("statement", ""),
                proof=item.get("proof", ""),
                latex_source=item.get("latex_source", ""),
            )
            theorems.append(thm)

        self.stats["total_theorems"] = len(theorems)
        logger.info(f"  提取到 {len(theorems)} 个定理")
        return theorems

    # ----------------------------------------------------------------
    # Step 5: 提取每个定理中的关键困难步骤
    # ----------------------------------------------------------------
    def step5_extract_key_steps(self, theorem: Theorem) -> List[KeyStep]:
        """提取定理证明中的每一个关键困难步骤"""
        logger.info(f"Step 5: 提取定理 {theorem.theorem_id} 的关键步骤...")

        system_prompt = "提示词5_system"
        # 提示词5: 你是一位数学证明分析专家。给定以下定理及其证明，
        # 请提取其中每一个关键困难步骤。关键困难步骤是指：
        # - 证明中具有本质数学困难的步骤
        # - 需要非平凡技巧或深刻洞察的步骤
        # - 不是简单的符号推导或直接引用已知结果的步骤
        # 以 JSON 数组格式输出：
        # [{"step_id": "Step 1", "description": "...", "proof_segment": "..."}, ...]

        user_prompt = (
            f"提示词5_user\n\n"
            f"定理ID: {theorem.theorem_id}\n"
            f"定理陈述: {theorem.statement}\n"
            f"证明: {theorem.proof}"
        )

        result = self.llm.call_json(system_prompt, user_prompt)

        steps = []
        for idx, item in enumerate(result):
            step = KeyStep(
                step_id=item.get("step_id", f"Step_{idx + 1}"),
                theorem_id=theorem.theorem_id,
                description=item.get("description", ""),
                proof_segment=item.get("proof_segment", ""),
            )
            steps.append(step)

        logger.info(f"  提取到 {len(steps)} 个关键步骤")
        return steps

    # ----------------------------------------------------------------
    # Step 6: 重要性审查
    # ----------------------------------------------------------------
    def step6_importance_review(self, step: KeyStep, theorem: Theorem) -> Tuple[bool, KeyStep]:
        """
        对关键步骤进行重要性评分。
        返回 (是否通过, 更新后的步骤)。
        若循环次数 >10 则报错作废。
        """
        logger.info(f"Step 6: 重要性审查 - {theorem.theorem_id} / {step.step_id}")

        for iteration in range(1, self.MAX_LOOP + 1):
            system_prompt = "提示词6_system"
            # 提示词6: 你是一位数学证明重要性评估专家。请对以下证明步骤进行重要性评分（0-100分）。
            # 评分标准：
            # - 该步骤是否包含本质数学困难
            # - 该步骤是否使用了非平凡的技巧
            # - 该步骤是否是证明的核心环节
            # 若评分不足95分，请说明原因并给出修改方案（如何重新划分或重新描述该步骤使其更聚焦于核心困难）。
            # 以 JSON 格式输出：
            # {"score": 整数, "reason": "...", "modification_plan": "...",
            #  "revised_description": "...", "revised_proof_segment": "..."}

            user_prompt = (
                f"提示词6_user\n\n"
                f"定理: {theorem.statement}\n"
                f"完整证明: {theorem.proof}\n\n"
                f"待评估步骤:\n"
                f"步骤ID: {step.step_id}\n"
                f"描述: {step.description}\n"
                f"证明片段: {step.proof_segment}\n\n"
                f"当前迭代次数: {iteration}/{self.MAX_LOOP}"
            )

            result = self.llm.call_json(system_prompt, user_prompt)
            score = int(result.get("score", 0))
            step.importance_score = score

            logger.info(f"  迭代 {iteration}: 重要性评分 = {score}")

            if score >= 95:
                return True, step

            # 评分不足95，根据修改方案更新步骤
            reason = result.get("reason", "")
            modification_plan = result.get("modification_plan", "")
            logger.info(f"  评分不足95。原因: {reason}")
            logger.info(f"  修改方案: {modification_plan}")

            revised_desc = result.get("revised_description", "")
            revised_segment = result.get("revised_proof_segment", "")

            if revised_desc:
                step.description = revised_desc
            if revised_segment:
                step.proof_segment = revised_segment

            # 返回 Step 5 重新提取（这里通过更新 step 内容来实现）

        # 超过最大循环次数，作废
        logger.error(f"  重要性审查超过 {self.MAX_LOOP} 次迭代，步骤作废: {step.step_id}")
        self.stats["discarded_steps"] += 1
        return False, step

    # ----------------------------------------------------------------
    # Step 7: 核心技巧分析
    # ----------------------------------------------------------------
    def step7_technique_analysis(self, step: KeyStep, theorem: Theorem) -> Idea:
        """询问证明中的核心技巧，补全对应的 idea"""
        logger.info(f"Step 7: 核心技巧分析 - {step.step_id}")

        system_prompt = "提示词7_system"
        # 提示词7: 你是一位数学证明技巧分析专家。请分析以下证明步骤中的核心技巧是什么，
        # 并补全它对应的 idea。核心技巧是指使该步骤得以成立的关键数学洞察或方法。
        # 以 JSON 格式输出：
        # {"technique_name": "...", "description": "技巧的完整描述",
        #  "idea": "该技巧背后的核心 idea", "details": "详细说明"}

        user_prompt = (
            f"提示词7_user\n\n"
            f"定理: {theorem.statement}\n"
            f"步骤描述: {step.description}\n"
            f"证明片段: {step.proof_segment}"
        )

        result = self.llm.call_json(system_prompt, user_prompt)

        idea = Idea(
            source="technique_analysis",
            description=result.get("idea", result.get("description", "")),
            details=result.get("details", ""),
        )

        logger.info(f"  技巧分析 idea: {idea.description[:80]}...")
        return idea

    # ----------------------------------------------------------------
    # Step 8: 人类思维路径分析
    # ----------------------------------------------------------------
    def step8_human_thinking_analysis(self, step: KeyStep, theorem: Theorem) -> Idea:
        """一步一步分析人类是如何想到这个证明的"""
        logger.info(f"Step 8: 人类思维路径分析 - {step.step_id}")

        system_prompt = "提示词8_system"
        # 提示词8: 你是一位数学教育专家和证明方法论专家。
        # 请一步一步分析这个证明步骤人类是如何想到的。
        # 每次找寻一个核心 idea，更新一次当前的证明状态。
        # 分析过程应包括：
        # 1. 初始状态（已知什么、要证什么）
        # 2. 每一步的思考过程（为什么想到用这个方法）
        # 3. 每一步之后证明状态的更新
        # 4. 最终提炼出的核心 idea
        # 以 JSON 格式输出：
        # {"thinking_steps": [{"step": 1, "idea": "...", "proof_state": "..."}, ...],
        #  "core_idea": "最终提炼的核心 idea",
        #  "details": "详细说明"}

        user_prompt = (
            f"提示词8_user\n\n"
            f"定理: {theorem.statement}\n"
            f"步骤描述: {step.description}\n"
            f"证明片段: {step.proof_segment}"
        )

        result = self.llm.call_json(system_prompt, user_prompt)

        idea = Idea(
            source="human_thinking",
            description=result.get("core_idea", ""),
            details=result.get("details", ""),
        )

        logger.info(f"  人类思维 idea: {idea.description[:80]}...")
        return idea

    # ----------------------------------------------------------------
    # Step 9: 综合比较选取更优 idea
    # ----------------------------------------------------------------
    def step9_compare_ideas(
        self, idea_technique: Idea, idea_human: Idea,
        step: KeyStep, theorem: Theorem
    ) -> Idea:
        """综合比较两种分析得到的 idea，选取更优的那一个"""
        logger.info(f"Step 9: 综合比较 ideas - {step.step_id}")

        system_prompt = "提示词9_system"
        # 提示词9: 你是一位数学方法论专家。请比较以下两个关于同一证明步骤的核心 idea 分析，
        # 选取更优的那一个。评判标准：
        # - 是否精准捕捉了证明的本质困难
        # - 是否简洁且具有启发性
        # - 是否可迁移到其他类似问题
        # 以 JSON 格式输出：
        # {"chosen": "technique_analysis" 或 "human_thinking",
        #  "reason": "选择理由",
        #  "final_idea": "最终选定的 idea 描述",
        #  "final_details": "最终选定的详细说明"}
        # 如果两者各有优点，可以综合两者给出一个更优的 idea。

        user_prompt = (
            f"提示词9_user\n\n"
            f"定理: {theorem.statement}\n"
            f"步骤: {step.description}\n\n"
            f"Idea A (技巧分析): {idea_technique.description}\n"
            f"详细: {idea_technique.details}\n\n"
            f"Idea B (人类思维分析): {idea_human.description}\n"
            f"详细: {idea_human.details}"
        )

        result = self.llm.call_json(system_prompt, user_prompt)

        chosen = result.get("chosen", "technique_analysis")
        best_idea = Idea(
            source=chosen,
            description=result.get("final_idea", ""),
            details=result.get("final_details", ""),
        )

        logger.info(f"  选定来源: {chosen}")
        return best_idea

    # ----------------------------------------------------------------
    # Step 10: 转化为 OG ID JSON 结构
    # ----------------------------------------------------------------
    def step10_convert_to_og_id(
        self, step: KeyStep, theorem: Theorem,
        global_assumptions: GlobalAssumptions, best_idea: Idea
    ) -> OgId:
        """将关键步骤及其 idea 转化为 og id 的 json struct 格式"""
        logger.info(f"Step 10: 转化为 OG ID - {step.step_id}")

        system_prompt = "提示词10_system"
        # 提示词10: 你是一位数学知识结构化专家。请将以下证明步骤转化为 OG ID 结构，包含：
        # ① global_assumptions: 全局假设（列表）
        # ② mathematical_objects: 数学对象（列表，如"有限群G"、"素数p"等）
        # ③ conditional_assumptions: 条件假设（列表，该步骤特有的前提条件）
        # ④ known_conclusions: 已知结论（列表，在该步骤之前已经建立的结论）
        # ⑤ proof_goal: 证明目标（字符串，该步骤要证明什么）
        # 要求：
        # - 每一项都应精简、准确、自包含
        # - 使用标准数学术语
        # - 证明目标应明确且具体
        # 以 JSON 格式输出。

        user_prompt = (
            f"提示词10_user\n\n"
            f"全局假设: {json.dumps(global_assumptions.assumptions, ensure_ascii=False)}\n"
            f"定理: {theorem.statement}\n"
            f"步骤描述: {step.description}\n"
            f"证明片段: {step.proof_segment}\n"
            f"核心 idea: {best_idea.description}\n"
            f"idea 详细: {best_idea.details}"
        )

        result = self.llm.call_json(system_prompt, user_prompt)

        og_id = OgId(
            og_id=f"{theorem.theorem_id}_{step.step_id}",
            global_assumptions=result.get("global_assumptions", global_assumptions.assumptions),
            mathematical_objects=result.get("mathematical_objects", []),
            conditional_assumptions=result.get("conditional_assumptions", []),
            known_conclusions=result.get("known_conclusions", []),
            proof_goal=result.get("proof_goal", ""),
            source_theorem_id=theorem.theorem_id,
            source_step_id=step.step_id,
        )

        return og_id

    # ----------------------------------------------------------------
    # Step 11: 简洁性核查
    # ----------------------------------------------------------------
    def step11_conciseness_check(self, og_id: OgId) -> Tuple[bool, int, str, str]:
        """
        对 og id 进行简洁性评分。
        返回 (是否通过, 分数, 原因, 修改方案)
        """
        system_prompt = "提示词11_system"
        # 提示词11: 你是一位数学知识简洁性评审专家。请对以下 OG ID 结构进行简洁性评分（0-100分）。
        # 评分标准：
        # - 是否有冗余信息
        # - 数学对象是否精简
        # - 条件假设是否最小化
        # - 证明目标是否简洁明确
        # 若不足95分，请说明原因及修改方案。
        # 以 JSON 格式输出：
        # {"score": 整数, "reason": "...", "modification_plan": "..."}

        user_prompt = (
            f"提示词11_user\n\n"
            f"OG ID:\n{json.dumps(asdict(og_id), ensure_ascii=False, indent=2)}"
        )

        result = self.llm.call_json(system_prompt, user_prompt)
        score = int(result.get("score", 0))
        reason = result.get("reason", "")
        plan = result.get("modification_plan", "")

        return score >= 95, score, reason, plan

    # ----------------------------------------------------------------
    # Step 12: 准确性核查
    # ----------------------------------------------------------------
    def step12_accuracy_check(self, og_id: OgId, step: KeyStep, theorem: Theorem) -> Tuple[bool, int, str, str]:
        """
        对 og id 进行准确性评分。
        返回 (是否通过, 分数, 原因, 修改方案)
        """
        system_prompt = "提示词12_system"
        # 提示词12: 你是一位严谨的数学准确性评审专家。请对以下 OG ID 结构进行准确性评分（0-100分）。
        # 评分标准：
        # - 全局假设是否准确反映原文
        # - 数学对象是否正确
        # - 条件假设是否完整且正确
        # - 已知结论是否确实在该步骤前已建立
        # - 证明目标是否准确对应原证明步骤的目标
        # 若不足99分，请说明原因及修改方案。
        # 以 JSON 格式输出：
        # {"score": 整数, "reason": "...", "modification_plan": "..."}

        user_prompt = (
            f"提示词12_user\n\n"
            f"原始定理: {theorem.statement}\n"
            f"原始证明: {theorem.proof}\n"
            f"原始步骤描述: {step.description}\n"
            f"原始证明片段: {step.proof_segment}\n\n"
            f"OG ID:\n{json.dumps(asdict(og_id), ensure_ascii=False, indent=2)}"
        )

        result = self.llm.call_json(system_prompt, user_prompt)
        score = int(result.get("score", 0))
        reason = result.get("reason", "")
        plan = result.get("modification_plan", "")

        return score >= 99, score, reason, plan

    # ----------------------------------------------------------------
    # Steps 7-12 组合: 对单个步骤执行完整的 idea 提取和质量核查
    # ----------------------------------------------------------------
    def process_single_step(
        self, step: KeyStep, theorem: Theorem,
        global_assumptions: GlobalAssumptions
    ) -> Optional[OgId]:
        """
        对单个关键步骤执行 Steps 7-12 的完整流程。
        包含简洁性和准确性的迭代核查。
        """
        logger.info(f"  处理步骤: {theorem.theorem_id} / {step.step_id}")

        for outer_iteration in range(1, self.MAX_LOOP + 1):
            logger.info(f"  外层迭代 {outer_iteration}/{self.MAX_LOOP}")

            # Step 7: 核心技巧分析
            idea_technique = self.step7_technique_analysis(step, theorem)

            # Step 8: 人类思维路径分析
            idea_human = self.step8_human_thinking_analysis(step, theorem)

            # Step 9: 综合比较
            best_idea = self.step9_compare_ideas(idea_technique, idea_human, step, theorem)

            # Step 10: 转化为 OG ID
            og_id = self.step10_convert_to_og_id(step, theorem, global_assumptions, best_idea)

            # Step 11: 简洁性核查
            concise_pass, concise_score, concise_reason, concise_plan = \
                self.step11_conciseness_check(og_id)
            og_id.conciseness_score = concise_score
            logger.info(f"  简洁性评分: {concise_score}")

            if not concise_pass:
                logger.info(f"  简洁性不足95。原因: {concise_reason}")
                logger.info(f"  修改方案: {concise_plan}")
                # 返回 Step 7 重新执行
                continue

            # Step 12: 准确性核查
            accuracy_pass, accuracy_score, accuracy_reason, accuracy_plan = \
                self.step12_accuracy_check(og_id, step, theorem)
            og_id.accuracy_score = accuracy_score
            logger.info(f"  准确性评分: {accuracy_score}")

            if not accuracy_pass:
                logger.info(f"  准确性不足99。原因: {accuracy_reason}")
                logger.info(f"  修改方案: {accuracy_plan}")
                # 返回 Step 7 重新执行
                continue

            # 两项核查均通过
            logger.info(f"  ✓ OG ID 通过所有核查: {og_id.og_id}")
            return og_id

        # 超过最大循环次数
        logger.error(f"  ✗ OG ID 质量核查超过 {self.MAX_LOOP} 次迭代，作废: {step.step_id}")
        self.stats["discarded_og_ids"] += 1
        return None

    # ----------------------------------------------------------------
    # 主流程
    # ----------------------------------------------------------------
    def run(self, latex_code: str) -> List[Dict[str, Any]]:
        """
        执行完整的 OG ID 提取流水线。

        参数:
            latex_code: 论文的 LaTeX 源代码

        返回:
            List[Dict]: 所有通过核查的 OG ID 列表
        """
        self.stats["start_time"] = time.time()
        logger.info("=" * 60)
        logger.info("OG ID 提取流水线启动")
        logger.info("=" * 60)

        # Step 2: 解析论文概览
        overview = self.step2_parse_overview(latex_code)

        # Step 3: 提取全局假设
        global_assumptions = self.step3_extract_global_assumptions(latex_code)

        # Step 4: 切分并提取定理
        theorems = self.step4_extract_theorems(latex_code)

        all_og_ids: List[OgId] = []

        for theorem in theorems:
            logger.info(f"\n{'─' * 40}")
            logger.info(f"处理定理: {theorem.theorem_id}")
            logger.info(f"{'─' * 40}")

            # Step 5: 提取关键困难步骤
            key_steps = self.step5_extract_key_steps(theorem)
            self.stats["total_steps"] += len(key_steps)

            for step in key_steps:
                # Step 6: 重要性审查
                passed, step = self.step6_importance_review(step, theorem)

                if not passed:
                    logger.warning(f"  步骤 {step.step_id} 未通过重要性审查，跳过")
                    continue

                # Steps 7-12: 对该步骤执行完整处理
                og_id = self.process_single_step(step, theorem, global_assumptions)

                if og_id is not None:
                    all_og_ids.append(og_id)

        # Step 13: 输出最终结果
        self.stats["end_time"] = time.time()
        self.stats["total_og_ids"] = len(all_og_ids)

        return self._finalize_output(all_og_ids, overview)

    # ----------------------------------------------------------------
    # 最终输出
    # ----------------------------------------------------------------
    def _finalize_output(
        self, og_ids: List[OgId], overview: PaperOverview
    ) -> List[Dict[str, Any]]:
        """Step 13: 格式化最终输出"""

        elapsed = self.stats["end_time"] - self.stats["start_time"]

        logger.info("\n" + "=" * 60)
        logger.info("OG ID 提取流水线完成")
        logger.info("=" * 60)
        logger.info(f"  论文研究领域: {overview.research_field}")
        logger.info(f"  提取定理数: {self.stats['total_theorems']}")
        logger.info(f"  提取关键步骤数: {self.stats['total_steps']}")
        logger.info(f"  作废步骤数: {self.stats['discarded_steps']}")
        logger.info(f"  成功生成 OG ID 数: {self.stats['total_og_ids']}")
        logger.info(f"  作废 OG ID 数: {self.stats['discarded_og_ids']}")
        logger.info(f"  总运行时间: {elapsed:.2f} 秒")
        logger.info("=" * 60)

        # 构建最终 JSON 输出
        output = []
        for og_id in og_ids:
            output.append({
                "og_id": og_id.og_id,
                "global_assumptions": og_id.global_assumptions,
                "mathematical_objects": og_id.mathematical_objects,
                "conditional_assumptions": og_id.conditional_assumptions,
                "known_conclusions": og_id.known_conclusions,
                "proof_goal": og_id.proof_goal,
                "metadata": {
                    "source_theorem_id": og_id.source_theorem_id,
                    "source_step_id": og_id.source_step_id,
                    "conciseness_score": og_id.conciseness_score,
                    "accuracy_score": og_id.accuracy_score,
                },
            })

        # 输出 JSON
        final_json = {
            "paper_overview": asdict(overview) if overview else {},
            "statistics": {
                "total_theorems": self.stats["total_theorems"],
                "total_key_steps": self.stats["total_steps"],
                "discarded_steps": self.stats["discarded_steps"],
                "total_og_ids": self.stats["total_og_ids"],
                "discarded_og_ids": self.stats["discarded_og_ids"],
                "runtime_seconds": round(elapsed, 2),
            },
            "og_ids": output,
        }

        # 保存到文件
        output_path = f"og_ids_output_{int(time.time())}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存至: {output_path}")

        return output


# ============================================================
# 批量处理多篇论文的入口
# ============================================================

class BatchProcessor:
    """批量处理多篇论文"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def process_papers(self, latex_codes: Dict[str, str]) -> Dict[str, List[Dict]]:
        """
        批量处理多篇论文

        参数:
            latex_codes: {arxiv_id: latex_code} 的字典

        返回:
            {arxiv_id: [og_ids]} 的字典
        """
        all_results = {}

        for arxiv_id, latex_code in latex_codes.items():
            logger.info(f"\n{'#' * 60}")
            logger.info(f"开始处理论文: {arxiv_id}")
            logger.info(f"{'#' * 60}")

            pipeline = OgIdPipeline(self.llm)

            try:
                og_ids = pipeline.run(latex_code)
                all_results[arxiv_id] = og_ids
            except Exception as e:
                logger.error(f"处理论文 {arxiv_id} 时出错: {e}")
                all_results[arxiv_id] = []

        # 汇总统计
        total_og_ids = sum(len(v) for v in all_results.values())
        logger.info(f"\n批量处理完成。共处理 {len(latex_codes)} 篇论文，"
                     f"生成 {total_og_ids} 个 OG ID。")

        # 保存汇总结果
        summary_path = f"batch_og_ids_{int(time.time())}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"汇总结果已保存至: {summary_path}")

        return all_results


# ============================================================
# 主入口
# ============================================================

def main():
    """主入口函数"""

    # ---- 配置 ----
    API_KEY = "your-api-key-here"
    MODEL = "gpt-4"
    BASE_URL = None  # 如有自定义端点则填入

    # 初始化 LLM 客户端
    llm = LLMClient(api_key=API_KEY, model=MODEL, base_url=BASE_URL)

    # ---- 假设已通过爬虫获取了论文 LaTeX 代码 ----
    # 这里用占位字符串代替，实际使用时替换为真实的 LaTeX 代码
    latex_codes = {
        "2401.00001": "% LaTeX code of paper 1 ...",
        "2401.00002": "% LaTeX code of paper 2 ...",
    }

    # ---- 单篇处理 ----
    # pipeline = OgIdPipeline(llm)
    # result = pipeline.run(latex_codes["2401.00001"])
    # print(json.dumps(result, ensure_ascii=False, indent=2))

    # ---- 批量处理 ----
    processor = BatchProcessor(llm)
    all_results = processor.process_papers(latex_codes)

    print(f"\n处理完成，共生成 {sum(len(v) for v in all_results.values())} 个 OG ID")


if __name__ == "__main__":
    main()
