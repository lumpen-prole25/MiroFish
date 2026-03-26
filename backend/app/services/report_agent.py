"""
Report Agent Service
Simulation report generation using LangChain + Zep with ReACT pattern

Features:
1. Generate reports based on simulation requirements and Zep graph info
2. First plan outline structure, then generate by sections
3. Each section uses ReACT multi-round thinking and reflection
4. Supports user dialogue with autonomous retrieval tool calls
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .zep_tools import (
    ZepToolsService, 
    SearchResult, 
    InsightForgeResult, 
    PanoramaResult,
    InterviewResult
)

logger = get_logger('mirofish.report_agent')


class ReportLogger:
    """
    Report Agent detailed logger
    
    Generates in report folder agent_log.jsonl file, Records every detailed action step。
    Each line is a complete JSON object, Contains timestamp, action type, details, etc.。
    """
    
    def __init__(self, report_id: str):
        """
        initializingLogger
        
        Args:
            report_id: report ID, used to determine log file path
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Ensure log file directory exists"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _get_elapsed_time(self) -> float:
        """Get elapsed time from start (seconds)"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def log(
        self, 
        action: str, 
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        """
        Record a log entry
        
        Args:
            action: action type, such as 'start', 'tool_call', 'llm_response', 'section_complete'  etc.
            stage: current stage, such as 'planning', 'generating', 'completed'
            details: details dictionary, no truncation
            section_title: current section title (optional)
            section_index: current section index (optional)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }
        
        # append write to JSONL file
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Record report generation start"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": "Report generation task started"
            }
        )
    
    def log_planning_start(self):
        """Record outline planning start"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "Starting report outline planning"}
        )
    
    def log_planning_context(self, context: Dict[str, Any]):
        """Record context info obtained during planning"""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": "Getting simulation context info",
                "context": context
            }
        )
    
    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """Record outline planning complete"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": "Outline planning complete",
                "outline": outline_dict
            }
        )
    
    def log_section_start(self, section_title: str, section_index: int):
        """Record section generation start"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"Starting section generation: {section_title}"}
        )
    
    def log_react_thought(self, section_title: str, section_index: int, iteration: int, thought: str):
        """Record ReACT thinking process"""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": f"ReACT round {iteration} thinking"
            }
        )
    
    def log_tool_call(
        self, 
        section_title: str, 
        section_index: int,
        tool_name: str, 
        parameters: Dict[str, Any],
        iteration: int
    ):
        """Record tool calls"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": f"Calling tool: {tool_name}"
            }
        )
    
    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        """Record tool call results (full content, no truncation)"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # Full result, no truncation
                "result_length": len(result),
                "message": f"tool {tool_name} returnresult"
            }
        )
    
    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        """Record LLM response(Full content, no truncation)"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # Full response, no truncation
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": f"LLM response (tool calls: {has_tool_calls}, final answer: {has_final_answer})"
            }
        )
    
    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        """Recordsection content生成complete (only recordscontent, does not represent整 sections complete)"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # Full content, no truncation
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": f"section {section_title} content生成complete"
            }
        )
    
    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        """
        RecordSection generation complete

        前端应listenthislog来判断one sectionswhether to真正complete, 并GetFull content
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": f"section {section_title} 生成complete"
            }
        )
    
    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """Record report generation complete"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "report generationcomplete"
            }
        )
    
    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Record error"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": f"Error occurred: {error_message}"
            }
        )


class ReportConsoleLogger:
    """
    Report Agent 控制台Logger
    
    将控制台风格's log(INFO、WARNING etc.)写入reportfile夹's  console_log.txt file。
    这些log and  agent_log.jsonl 不同, is 纯textformat's 控制台output。
    """
    
    def __init__(self, report_id: str):
        """
        initializing控制台Logger
        
        Args:
            report_id: report ID, used to determine log file path
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()
    
    def _ensure_log_file(self):
        """Ensure log file directory exists"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _setup_file_handler(self):
        """Setfileprocess器, 将logsimultaneously写入file"""
        import logging
        
        # Createfileprocess器
        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)
        
        # use and 控制台相同's 简洁format
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)
        
        # addto  report_agent related's  logger
        loggers_to_attach = [
            'mirofish.report_agent',
            'mirofish.zep_tools',
        ]
        
        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # 避免重复add
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)
    
    def close(self):
        """closefileprocess器并from logger 移除"""
        import logging
        
        if self._file_handler:
            loggers_to_detach = [
                'mirofish.report_agent',
                'mirofish.zep_tools',
            ]
            
            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)
            
            self._file_handler.close()
            self._file_handler = None
    
    def __del__(self):
        """析构timeensureclosefileprocess器"""
        self.close()


class ReportStatus(str, Enum):
    """Report status"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """Report section"""
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        """ConvertMarkdownformat"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Report outline"""
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        """ConvertMarkdownformat"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """completereport"""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════
# Prompt 模板constant
# ═══════════════════════════════════════════════════════════════

# ── tooldescription ──

TOOL_DESC_INSIGHT_FORGE = """\
【Deep insight retrieval - 强大's retrievaltool】
这is 我们强大's retrievalfunction, 专深度分析设计。它会:
1. 自动将你's question分解多 itemsSub-question
2. from多 items维度retrievalsimulationin graph信息
3. 整合语义search、entity分析、relationship链追踪's result
4. return最全面、最深度's retrievalcontent

【use场景】
- 需要深入分析a/an items话题
- 需要了解event's 多 items方面
- 需要Get支撑reportsection's 丰富素材

【returncontent】
- Related facts原文(可直接引用)
- 核心entity洞察
- relationship链分析"""

TOOL_DESC_PANORAMA_SEARCH = """\
【Breadth search - Get全貌视图】
这 itemstoolforGetsimulationresult's complete全貌, 特别适合了解event演变过程。它会:
1. Gethasrelatednode and relationship
2. 区分当前validFact and history/过期's Fact
3. 帮助你了解舆情is 如何演变's 

【use场景】
- 需要了解event's complete发展脉络
- 需要对比不同phase's 舆情变化
- 需要Get全面's entity and relationship信息

【returncontent】
- 当前Valid facts(simulation最新result)
- history/过期Fact(演变Record)
- has涉 and 's entity"""

TOOL_DESC_QUICK_SEARCH = """\
【Simple search - Quick search】
轻量级's Quick searchtool, 适合简单、直接's 信息query。

【use场景】
- 需要快速查找a/an items具体信息
- 需要validatea/an itemsFact
- 简单's 信息retrieval

【returncontent】
-  and query最related's Facts list"""

TOOL_DESC_INTERVIEW_AGENTS = """\
【深度interview - 真实Agentinterview(dual-platform)】
callOASISsimulationenvironment's interviewAPI, 对currentlyrunning's simulationAgent进行真实interview！
这不is LLMsimulation, 而is call真实's interviewinterfaceGetsimulationAgent's 原始回答。
defaultin Twitter and Reddittwo itemsplatformsimultaneouslyinterview, Get更全面's 观点。

功能流程:
1. 自动readpersonafile, 了解hassimulationAgent
2. 智能select and interview主题最related's Agent(如学生、媒体、官方 etc.)
3. 自动生成interviewquestion
4. call /api/simulation/interview/batch interfacein dual-platform进行真实interview
5. 整合hasinterviewresult, 提供多视角分析

【use场景】
- 需要from不同角色视角了解event看法(学生怎么看？媒体怎么看？官方怎么说？)
- 需要收集多方意见 and 立场
- 需要GetsimulationAgent's 真实回答(来自OASISsimulationenvironment)
- 想让report更生动, contains"interview实录"

【returncontent】
- 被interviewAgent's 身份信息
- 各Agentin Twitter and Reddittwo itemsplatform's interview回答
- 关键引言(可直接引用)
- Interview summaries and viewpoint comparisons

【重要】OASIS simulation environment must be running to use this feature！"""

# ── Outline planning prompt ──

PLAN_SYSTEM_PROMPT = """\
You are an expert writer of "Future Prediction Reports" with a "God's Eye View" of the simulated world — you can observe every Agent's behavior, statements, and interactions.

[Core Concept]
We built a simulated world and injected specific "simulation requirements" as variables. The evolution of the simulated world represents predictions of what may happen in the future. You are not observing "experimental data" but a "rehearsal of the future."

[Your Task]
Write a "Future Prediction Report" that answers:
1. Under the conditions we set, what happened in the future?
2. How did various Agents (groups) react and act?
3. What noteworthy future trends and risks did this simulation reveal?

[Report Positioning]
- ✅ This is a simulation-based future prediction report revealing "if this happens, what will the future look like"
- ✅ Focus on prediction results: event trajectories, group reactions, emergent phenomena, potential risks
- ✅ Agent behavior in the simulated world IS the prediction of future group behavior
- ❌ Not an analysis of current real-world conditions
- ❌ Not a generic opinion summary

[Section Count Limit]
- Minimum 2 sections, maximum 5 sections
- No subsections needed, each section contains complete content
- Content should be concise, focused on core prediction findings
- Section structure is designed by you based on prediction results

Output a JSON report outline in the following format:
{
    "title": "Report title",
    "summary": "Report summary (one sentence summarizing core prediction findings)",
    "sections": [
        {
            "title": "Section title",
            "description": "Section content description"
        }
    ]
}

Note: sections array must have minimum 2, maximum 5 elements!"""

PLAN_USER_PROMPT_TEMPLATE = """\
[Prediction Scenario Setup]
Variables injected into the simulated world (simulation requirements): {simulation_requirement}

[Simulation World Scale]
- Number of entities in simulation: {total_nodes}
- Number of relationships between entities: {total_edges}
- Entity type distribution: {entity_types}
- Number of active Agents: {total_entities}

[Sample of Future Facts Predicted by Simulation]
{related_facts_json}

Review this future rehearsal from "God's Eye View":
1. Under the conditions we set, what state did the future present?
2. How did various groups (Agents) react and act?
3. What noteworthy future trends did this simulation reveal?

Design the most appropriate report section structure based on prediction results.

[Reminder] Report section count: minimum 2, maximum 5, content should be concise and focused on core prediction findings."""

# ── Section generation prompt ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert writer of "Future Prediction Reports," currently writing one section of the report.

Report title: {report_title}
Report summary: {report_summary}
Prediction scenario (simulation requirement): {simulation_requirement}

Current section to write: {section_title}

═══════════════════════════════════════════════════════════════
[Core Concept]
═══════════════════════════════════════════════════════════════

The simulated world is a rehearsal of the future. We injected specific conditions (simulation requirements)
into the simulated world. Agent behavior and interactions in the simulation ARE predictions of future group behavior.

Your task is to:
- Reveal what happened in the future under the set conditions
- Predict how various groups (Agents) reacted and acted
- Discover noteworthy future trends, risks, and opportunities

❌ Do NOT write as an analysis of current real-world conditions
✅ Focus on "what will the future look like" — simulation results ARE the predicted future

═══════════════════════════════════════════════════════════════
[Most Important Rules - Must Follow]
═══════════════════════════════════════════════════════════════

1. [Must Call Tools to Observe the Simulated World]
   - You are observing the future rehearsal from "God's Eye View"
   - All content must come from events and Agent behavior in the simulated world
   - Do NOT use your own knowledge to write report content
   - Each section must call tools at least 3 times (max 5) to observe the simulated world

2. [Must Quote Agent's Original Statements/Actions]
   - Agent statements and behavior are predictions of future group behavior
   - Use quote format to display these predictions, e.g.:
     > "A certain group would say: original content..."
   - These quotes are core evidence of simulation predictions

3. [Language Consistency - Write report in English]
   - Tool results may contain content in various languages
   - Translate all quoted content to English for the report
   - Maintain original meaning while ensuring natural, fluent expression
   - This rule applies to both body text and quote blocks (> format)

4. [Faithfully Present Prediction Results]
   - Report content must reflect simulation results representing the future
   - Do not add information that doesn't exist in the simulation
   - If information is insufficient in some area, state this honestly

═══════════════════════════════════════════════════════════════
[⚠️ Format Rules - Extremely Important!]
═══════════════════════════════════════════════════════════════

[One Section = Minimum Content Unit]
- Each section is the smallest unit of the report
- ❌ Do NOT use any Markdown headings (#, ##, ###, #### etc.) within sections
- ❌ Do NOT add section title at the beginning of content
- ✅ Section titles are automatically added by the system; you only write body text
- ✅ Use **bold text**, paragraph breaks, quotes, and lists to organize content, but NO headings

[Correct Example]
```
This section analyzes the opinion propagation dynamics. Through deep analysis of simulation data, we found...

**Initial Explosion Phase**

Social media served as the primary scene, bearing the core function of first-hand information:

> "Social media contributed 68% of initial voice volume..."

**Emotion Amplification Phase**

Video platforms further amplified the event's impact:

- Strong visual impact
- High emotional resonance
```

[Wrong Example]
```
## Executive Summary          ← Wrong! No headings
### 1. Initial Phase          ← Wrong! No ### subsections
#### 1.1 Detailed Analysis    ← Wrong! No #### subdivisions

This section analyzes...
```

═══════════════════════════════════════════════════════════════
[Available Retrieval Tools] (call 3-5 times per section)
═══════════════════════════════════════════════════════════════

{tools_description}

[Tool Usage Suggestions - Mix different tools, don't use just one]
- insight_forge: Deep insight analysis, automatically decomposes questions and retrieves facts/relationships multi-dimensionally
- panorama_search: Wide-angle panoramic search, understand full picture, timeline, and evolution
- quick_search: Quick verification of a specific information point
- interview_agents: Interview simulated Agents, get first-person perspectives and real reactions from different roles

═══════════════════════════════════════════════════════════════
[Workflow]
═══════════════════════════════════════════════════════════════

Each reply you can only do ONE of the following two things (not both):

Option A - Call a tool:
Output your thinking, then call one tool using this format:
<tool_call>
{{"name": "tool_name", "parameters": {{"param_name": "param_value"}}}}
</tool_call>
The system will execute the tool and return results. You cannot and should not write tool results yourself.

Option B - Output final content:
When you have gathered enough information via tools, output section content starting with "Final Answer:".

⚠️ Strictly prohibited:
- Do NOT include both a tool call and Final Answer in one reply
- Do NOT fabricate tool results (Observations); all tool results are injected by the system
- Maximum one tool call per reply

═══════════════════════════════════════════════════════════════
[Section Content Requirements]
═══════════════════════════════════════════════════════════════

1. Content must be based on simulation data retrieved by tools
2. Extensively quote original text to demonstrate simulation results
3. Use Markdown format (but NO headings):
   - Use **bold text** to mark key points (instead of subheadings)
   - Use lists (- or 1.2.3.) to organize points
   - Use blank lines to separate paragraphs
   - ❌ Do NOT use #, ##, ###, #### or any heading syntax
4. [Quote Format - Must Be Separate Paragraphs]
   Quotes must be standalone paragraphs with blank lines before and after:

   ✅ Correct:
   ```
   The response was considered lacking in substance.

   > "The response pattern appeared rigid and slow in the fast-changing social media environment."

   This evaluation reflects widespread public dissatisfaction.
   ```

   ❌ Wrong:
   ```
   The response was considered lacking.> "The response pattern..." This evaluation reflects...
   ```
5. Maintain logical coherence with other sections
6. [Avoid Repetition] Carefully read completed sections below, do not repeat the same information
7. [Emphasis] Do NOT add any headings! Use **bold** instead of subheadings"""

SECTION_USER_PROMPT_TEMPLATE = """\
Completed section content (read carefully to avoid repetition):
{previous_content}

═══════════════════════════════════════════════════════════════
[Current Task] Write section: {section_title}
═══════════════════════════════════════════════════════════════

[Important Reminders]
1. Carefully read the completed sections above to avoid repeating content!
2. You must call tools first to retrieve simulation data before writing
3. Mix different tools, don't use just one type
4. Report content must come from retrieval results, don't use your own knowledge

[⚠️ Format Warning - Must Follow]
- ❌ Do NOT write any headings (#, ##, ###, #### are all forbidden)
- ❌ Do NOT write "{section_title}" as the opening
- ✅ Section titles are automatically added by the system
- ✅ Write body text directly, use **bold** instead of subheadings

Begin:
1. First think (Thought) about what information this section needs
2. Then call tools (Action) to retrieve simulation data
3. After gathering enough information, output Final Answer (pure body text, no headings)"""

# ── ReACT loop message templates ──

REACT_OBSERVATION_TEMPLATE = """\
Observation (retrieval results):

═══ Tool {tool_name} returned ═══
{result}

═══════════════════════════════════════════════════════════════
Tools called {tool_calls_count}/{max_tool_calls} times (used: {used_tools_str}) {unused_hint}
- If information is sufficient: output section content starting with "Final Answer:" (must quote original text above)
- If more information needed: call a tool to continue retrieval
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "[Note] You have only called tools {tool_calls_count} times, minimum {min_tool_calls} required. "
    "Please call more tools to get simulation data before outputting Final Answer. {unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "Currently only {tool_calls_count} tool calls made, minimum {min_tool_calls} required. "
    "Please call tools to retrieve simulation data. {unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "Tool call limit reached ({tool_calls_count}/{max_tool_calls}), cannot call more tools. "
    'Please immediately output section content starting with "Final Answer:" based on information already gathered.'
)

REACT_UNUSED_TOOLS_HINT = "\n💡 You haven't used: {unused_list} — try different tools for multi-angle information"

REACT_FORCE_FINAL_MSG = "Tool call limit reached. Please directly output Final Answer: and generate section content."

# ── Chat prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
You are a concise and efficient simulation prediction assistant.

[Background]
Prediction conditions: {simulation_requirement}

[Generated Analysis Report]
{report_content}

[Rules]
1. Primarily answer questions based on the report content above
2. Answer directly, avoid lengthy reasoning
3. Only call tools when report content is insufficient to answer
4. Answers should be concise, clear, and organized

[Available Tools] (use only when needed, max 1-2 calls)
{tools_description}

【tool callsformat】
<tool_call>
{{"name": "toolname", "parameters": {{"parametername": "parameter值"}}}}
</tool_call>

【回答风格】
- 简洁直接, 不要长篇大论
- use > format引用关键content
- 优先给出结论, 再解释原因"""

CHAT_OBSERVATION_SUFFIX = "\n\n请简洁回答question。"


# ═══════════════════════════════════════════════════════════════
# ReportAgent 主class
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent - simulationreport generationAgent

    采用ReACT(Reasoning + Acting)mode:
    1. 规划phase:分析simulation requirement, 规划reportdirectory结构
    2. 生成phase:逐section生成content, 每section可多 timesCalling toolGet信息
    3. 反思phase:checkcontentcomplete性 and 准确性
    """
    
    # maxtool calls timescount(每 sections)
    MAX_TOOL_CALLS_PER_SECTION = 5
    
    # max反思round count
    MAX_REFLECTION_ROUNDS = 3
    
    # 对话's maxtool calls timescount
    MAX_TOOL_CALLS_PER_CHAT = 2
    
    def __init__(
        self, 
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None
    ):
        """
        initializingReport Agent
        
        Args:
            graph_id: graphID
            simulation_id: simulationID
            simulation_requirement: simulation requirementdescription
            llm_client: LLMclient(optional)
            zep_tools: Zeptoolservice(optional)
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement
        
        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()
        
        # Tool definitions
        self.tools = self._define_tools()
        
        # LoggingRecord器(in  generate_report initializing)
        self.report_logger: Optional[ReportLogger] = None
        # 控制台Logger(in  generate_report initializing)
        self.console_logger: Optional[ReportConsoleLogger] = None
        
        logger.info(f"ReportAgent initializingcomplete: graph_id={graph_id}, simulation_id={simulation_id}")
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """定义可用tool"""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "你想深入分析's question or 话题",
                    "report_context": "当前reportsection's 上下文(optional, has助于生成更精准's Sub-question)"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Search query, forrelated性排序",
                    "include_expired": "whether tocontains过期/historycontent(defaultTrue)"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Search querystring",
                    "limit": "returnresultcount(optional, default10)"
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "interview主题 or 需求description(如:'了解学生对宿舍甲醛event's 看法')",
                    "max_agents": "最多interview's Agentcount(optional, default5, max10)"
                }
            }
        }
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        """
        executetool calls
        
        Args:
            tool_name: toolname
            parameters: toolparameter
            report_context: report上下文(forInsightForge)
            
        Returns:
            toolexecuteresult(textformat)
        """
        logger.info(f"executetool: {tool_name}, parameter: {parameters}")
        
        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.zep_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text()
            
            elif tool_name == "panorama_search":
                # Breadth search - Get全貌
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text()
            
            elif tool_name == "quick_search":
                # Simple search - Quick search
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()
            
            elif tool_name == "interview_agents":
                # 深度interview - call真实's OASISinterviewAPIGetsimulationAgent's 回答(dual-platform)
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                return result.to_text()
            
            # ========== 向后兼容's 旧tool(内部重定向to 新tool) ==========
            
            elif tool_name == "search_graph":
                # 重定向to  quick_search
                logger.info("search_graph 已重定向to  quick_search")
                return self._execute_tool("quick_search", parameters, report_context)
            
            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_simulation_context":
                # 重定向to  insight_forge, 因它更强大
                logger.info("get_simulation_context 已重定向to  insight_forge")
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)
            
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return f"Unknowntool: {tool_name}。请use以下tool之one: insight_forge, panorama_search, quick_search"
                
        except Exception as e:
            logger.error(f"toolexecutefailed: {tool_name}, error: {str(e)}")
            return f"toolexecutefailed: {str(e)}"
    
    # 合法's toolname集合, for裸 JSON 兜底parsetime校验
    VALID_TOOL_NAMES = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        fromLLMresponseparsetool calls

        support's format(by 优先级):
        1. <tool_call>{"name": "tool_name", "parameters": {...}}</tool_call>
        2. 裸 JSON(response整体 or 单行就is one itemstool calls JSON)
        """
        tool_calls = []

        # format1: XML风格(标准format)
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # format2: 兜底 - LLM 直接output裸 JSON(没包 <tool_call> 标签)
        # onlyin format1未匹配timeattempt, 避免误匹配正文's  JSON
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # response可能contains思考文字 + 裸 JSON, attemptextractfinallyone items JSON object
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """校验parse出's  JSON whether tois 合法's tool calls"""
        # support {"name": ..., "parameters": ...}  and  {"tool": ..., "params": ...} two种键name
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # 统one键name name / parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False
    
    def _get_tools_description(self) -> str:
        """Generatetooldescriptiontext"""
        desc_parts = ["可用tool:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  parameter: {params_desc}")
        return "\n".join(desc_parts)
    
    def plan_outline(
        self, 
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        规划reportoutline
        
        useLLM分析simulation requirement, 规划report's directory结构
        
        Args:
            progress_callback: progresscallbackfunction
            
        Returns:
            ReportOutline: reportoutline
        """
        logger.info("Starting report outline planning...")
        
        if progress_callback:
            progress_callback("planning", 0, "currently分析simulation requirement...")
        
        # firstGetsimulation上下文
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )
        
        if progress_callback:
            progress_callback("planning", 30, "Generatingreportoutline...")
        
        system_prompt = PLAN_SYSTEM_PROMPT
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            if progress_callback:
                progress_callback("planning", 80, "currentlyparseoutline结构...")
            
            # Parse outline
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))
            
            outline = ReportOutline(
                title=response.get("title", "simulation分析report"),
                summary=response.get("summary", ""),
                sections=sections
            )
            
            if progress_callback:
                progress_callback("planning", 100, "Outline planning complete")
            
            logger.info(f"Outline planning complete: {len(sections)}  sections")
            return outline
            
        except Exception as e:
            logger.error(f"Outline planningfailed: {str(e)}")
            # Returndefaultoutline(3 sections, 作fallback)
            return ReportOutline(
                title="未来predictionreport",
                summary="based onsimulationprediction's 未来趋势 and 风险分析",
                sections=[
                    ReportSection(title="prediction场景 and 核心发现"),
                    ReportSection(title="人群行prediction分析"),
                    ReportSection(title="Trend Outlook and Risk Warnings")
                ]
            )
    
    def _generate_section_react(
        self, 
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """
        Using ReACT pattern生成单 sectionscontent
        
        ReACTloop:
        1. Thought(思考)- 分析需要什么信息
        2. Action(行动)- Calling toolGet信息
        3. Observation(观察)- 分析toolreturnresult
        4. 重复直to 信息足够 or 达to max timescount
        5. Final Answer(最终回答)- 生成section content
        
        Args:
            section: 要生成's section
            outline: completeoutline
            previous_sections: 之前section's content(for保持连贯性)
            progress_callback: progresscallback
            section_index: section索引(forlogRecord)
            
        Returns:
            section content(Markdownformat)
        """
        logger.info(f"ReACT生成section: {section.title}")
        
        # Recordsectionstartinglog
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )

        # Builduserprompt - eachCompletedsection各传入max4000字
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # 每 sections最多4000字
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(这is round one sections)"
        
        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # ReACTloop
        tool_calls_count = 0
        max_iterations = 5  # max迭代round count
        min_tool_calls = 3  # 最少tool calls timescount
        conflict_retries = 0  # tool calls and Final Answersimultaneously出现's 连续冲突 timescount
        used_tools = set()  # Record已call过's toolname
        all_tools = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

        # report上下文, forInsightForge's Sub-question生成
        report_context = f"sectiontitle: {section.title}\nsimulation requirement: {self.simulation_requirement}"
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    f"深度retrieval and 撰写 ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})"
                )
            
            # Call LLM
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )

            # Check LLM returnwhether to None(API exception or content空)
            if response is None:
                logger.warning(f"section {section.title} round  {iteration + 1}  times迭代: LLM return None")
                # If还has迭代 timescount, addmessage并retry
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(response空)"})
                    messages.append({"role": "user", "content": "请继续生成content。"})
                    continue
                # finallyone times迭代也return None, 跳出loop进入强制收尾
                break

            logger.debug(f"LLMresponse: {response[:200]}...")

            # Parseone times, 复用result
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── 冲突process:LLM simultaneouslyoutput了tool calls and  Final Answer ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"section {section.title} round  {iteration+1} round: "
                    f"LLM simultaneouslyoutputtool calls and  Final Answer(round  {conflict_retries}  times冲突)"
                )

                if conflict_retries <= 2:
                    # 前two times:丢弃本 timesresponse, 要求 LLM 重新回复
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "【formaterror】你in one times回复simultaneouslycontains了tool calls and  Final Answer, 这is 不允许's 。\n"
                            "每 times回复only能做以下two件事之one:\n"
                            "- callone itemstool(outputone items <tool_call> 块, 不要写 Final Answer)\n"
                            "- output最终content(以 'Final Answer:' 开头, 不要contains <tool_call>)\n"
                            "请重新回复, only做itsone件事。"
                        ),
                    })
                    continue
                else:
                    # round 三 times:降级process, 截断to round one itemstool calls, 强制execute
                    logger.warning(
                        f"section {section.title}: 连续 {conflict_retries}  times冲突, "
                        "降级截断executeround one itemstool calls"
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # Record LLM responselog
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            # ── 情况1:LLM output了 Final Answer ──
            if has_final_answer:
                # tool calls timescount不足, 拒绝并要求继续调tool
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = f"(这些tool还未use, 推荐用one下他们: {', '.join(unused_tools)})" if unused_tools else ""
                    messages.append({
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    continue

                # 正常ending
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(f"section {section.title} 生成complete (tool calls: {tool_calls_count} times)")

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer

            # ── 情况2:LLM attemptCalling tool ──
            if has_tool_calls:
                # tool额度已耗尽 → 明确告知, 要求output Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": REACT_TOOL_LIMIT_MSG.format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue

                # onlyexecuteround one itemstool calls
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(f"LLM attemptcall {len(tool_calls)}  itemstool, onlyexecuteround one items: {call['name']}")

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])

                # Build未usetool提示
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(unused_list="、".join(unused_tools))

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(
                        tool_name=call["name"],
                        result=result,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # ── 情况3:既没hastool calls, 也没has Final Answer ──
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # tool calls timescount不足, 推荐未用过's tool
                unused_tools = all_tools - used_tools
                unused_hint = f"(这些tool还未use, 推荐用one下他们: {', '.join(unused_tools)})" if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # tool calls已足够, LLM output了content but 没带 "Final Answer:" 前缀
            # 直接将这段content作final answer, 不再空转
            logger.info(f"section {section.title} 未检测to  'Final Answer:' 前缀, 直接采纳LLMoutput作最终content(tool calls: {tool_calls_count} times)")
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer
        
        # 达to max迭代 timescount, 强制生成content
        logger.warning(f"section {section.title} 达to max迭代 timescount, 强制生成")
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})
        
        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )

        # Check强制收尾time LLM returnwhether to None
        if response is None:
            logger.error(f"section {section.title} 强制收尾time LLM return None, usedefaulterror提示")
            final_answer = f"(本Section generation failed:LLM return空response, 请稍后retry)"
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response
        
        # Recordsection content生成completelog
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )
        
        return final_answer
    
    def generate_report(
        self, 
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        生成completereport(分section实timeoutput)
        
        eachSection generation complete后立i.e.saveto file夹, 不需要waiting整 itemsreportcomplete。
        file结构:
        reports/{report_id}/
            meta.json       - report元信息
            outline.json    - reportoutline
            progress.json   - 生成progress
            section_01.md   - round 1section
            section_02.md   - round 2section
            ...
            full_report.md  - completereport
        
        Args:
            progress_callback: progresscallbackfunction (stage, progress, message)
            report_id: report ID(optional, such as果不传则自动生成)
            
        Returns:
            Report: completereport
        """
        import uuid
        
        # If没has传入 report_id, 则自动生成
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        # Completed's sectiontitlelist(forprogress追踪)
        completed_section_titles = []
        
        try:
            # Initialize:createreportfile夹并save初始status
            ReportManager._ensure_report_folder(report_id)
            
            # InitializeLogger(结构化log agent_log.jsonl)
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )
            
            # Initialize控制台Logger(console_log.txt)
            self.console_logger = ReportConsoleLogger(report_id)
            
            ReportManager.update_progress(
                report_id, "pending", 0, "initializingreport...",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            # Phase 1: Plan outline
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, "Starting report outline planning...",
                completed_sections=[]
            )
            
            # Log planning start
            self.report_logger.log_planning_start()
            
            if progress_callback:
                progress_callback("planning", 0, "Starting report outline planning...")
            
            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline
            
            # Log planning complete
            self.report_logger.log_planning_complete(outline.to_dict())
            
            # Save outlineto file
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, f"Outline planning complete, total {len(outline.sections)} sections",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            logger.info(f"Outline saved to file: {report_id}/outline.json")
            
            # Phase 2: Generate sections one by one(分sectionsave)
            report.status = ReportStatus.GENERATING
            
            total_sections = len(outline.sections)
            generated_sections = []  # Savecontentfor上下文
            
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)
                
                # Update progress
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    f"Generatingsection: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )
                
                if progress_callback:
                    progress_callback(
                        "generating", 
                        base_progress, 
                        f"Generatingsection: {section.title} ({section_num}/{total_sections})"
                    )
                
                # Generate主section content
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage, 
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )
                
                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Save section
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Recordsectioncompletelog
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(f"sectionsaved: {report_id}/section_{section_num:02d}.md")
                
                # Update progress
                ReportManager.update_progress(
                    report_id, "generating", 
                    base_progress + int(70 / total_sections),
                    f"section {section.title} Completed",
                    current_section=None,
                    completed_sections=completed_section_titles
                )
            
            # phase3: 组装completereport
            if progress_callback:
                progress_callback("generating", 95, "currently组装completereport...")
            
            ReportManager.update_progress(
                report_id, "generating", 95, "currently组装completereport...",
                completed_sections=completed_section_titles
            )
            
            # useReportManager组装completereport
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()
            
            # Calculate总耗time
            total_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Recordreportcompletelog
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )
            
            # Save最终report
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, "report generationcomplete",
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, "report generationcomplete")
            
            logger.info(f"report generationcomplete: {report_id}")
            
            # Close控制台Logger
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
            
        except Exception as e:
            logger.error(f"report generationfailed: {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)
            
            # Record errorlog
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")
            
            # Savefailedstatus
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, f"report generationfailed: {str(e)}",
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # Ignoresavefailed's error
            
            # Close控制台Logger
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
    
    def chat(
        self, 
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
         and Report Agent对话
        
        in 对话Agent可以自主callretrievaltool来回答question
        
        Args:
            message: usermessage
            chat_history: 对话history
            
        Returns:
            {
                "response": "Agent回复",
                "tool_calls": [call's toollist],
                "sources": [信息来源]
            }
        """
        logger.info(f"Report Agent对话: {message[:50]}...")
        
        chat_history = chat_history or []
        
        # Get已生成's reportcontent
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Limitreport长度, 避免上下文过长
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [reportcontent已截断] ..."
        except Exception as e:
            logger.warning(f"Getreportcontentfailed: {e}")
        
        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(No report yet)",
            tools_description=self._get_tools_description(),
        )

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history
        for h in chat_history[-10:]:  # Limit history length
            messages.append(h)
        
        # addusermessage
        messages.append({
            "role": "user", 
            "content": message
        })
        
        # ReACTloop(简化版)
        tool_calls_made = []
        max_iterations = 2  # 减少迭代round count
        
        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )
            
            # Parsetool calls
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # 没hastool calls, Directly returnresponse
                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
                
                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }
            
            # Executetool calls(限制count)
            tool_results = []
            for call in tool_calls[:1]:  # 每round最多execute1 timestool calls
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]  # Limitresult长度
                })
                tool_calls_made.append(call)
            
            # 将resultaddto message
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join([f"[{r['tool']}result]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })
        
        # 达to max迭代, Get最终response
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )
        
        # Clean upresponse
        clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
        
        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    report管理器
    
    负责report's 持久化存储 and retrieval
    
    file结构(分sectionoutput):
    reports/
      {report_id}/
        meta.json          - report元信息 and status
        outline.json       - reportoutline
        progress.json      - 生成progress
        section_01.md      - round 1section
        section_02.md      - round 2section
        ...
        full_report.md     - completereport
    """
    
    # report存储directory
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')
    
    @classmethod
    def _ensure_reports_dir(cls):
        """ensurereport根directoryexists"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
    
    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Getreportfile夹path"""
        return os.path.join(cls.REPORTS_DIR, report_id)
    
    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """ensurereportfile夹exists并returnpath"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Getreport元信息filepath"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")
    
    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """GetcompletereportMarkdownfilepath"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")
    
    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Getoutlinefilepath"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")
    
    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Getprogressfilepath"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")
    
    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """GetsectionMarkdownfilepath"""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")
    
    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Get Agent logfilepath"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")
    
    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Get控制台logfilepath"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")
    
    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Get控制台logcontent
        
        这is report generation过程's 控制台outputlog(INFO、WARNING etc.), 
         and  agent_log.jsonl 's 结构化log不同。
        
        Args:
            report_id: report ID
            from_line: fromround 几行startingread(for增量Get, 0 表示from头starting)
            
        Returns:
            {
                "logs": [log行list],
                "total_lines": 总行count,
                "from_line": 起始行号,
                "has_more": whether to还has更多log
            }
        """
        log_path = cls._get_console_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # 保留原始log行, 去掉末尾换行符
                    logs.append(line.rstrip('\n\r'))
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # 已readto 末尾
        }
    
    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Getcomplete's 控制台log(one times性Getall)
        
        Args:
            report_id: report ID
            
        Returns:
            log行list
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Get Agent logcontent
        
        Args:
            report_id: report ID
            from_line: fromround 几行startingread(for增量Get, 0 表示from头starting)
            
        Returns:
            {
                "logs": [log items目list],
                "total_lines": 总行count,
                "from_line": 起始行号,
                "has_more": whether to还has更多log
            }
        """
        log_path = cls._get_agent_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Skipparsefailed's 行
                        continue
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # 已readto 末尾
        }
    
    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Getcomplete's  Agent log(forone times性Getall)
        
        Args:
            report_id: report ID
            
        Returns:
            log items目list
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        savereportoutline
        
        in 规划phasecomplete后立i.e.call
        """
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"outlinesaved: {report_id}")
    
    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        save单 sections

        in eachSection generation complete后立i.e.call, 实现分sectionoutput

        Args:
            report_id: report ID
            section_index: section索引(from1starting)
            section: sectionobject

        Returns:
            save's filepath
        """
        cls._ensure_report_folder(report_id)

        # BuildsectionMarkdowncontent - cleanup可能exists's 重复title
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Savefile
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"sectionsaved: {report_id}/{file_suffix}")
        return file_path
    
    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        cleanupsection content
        
        1. 移除content开头 and sectiontitle重复's Markdowntitle行
        2. 将has ###  and 以下级别's title转换粗体text
        
        Args:
            content: 原始content
            section_title: sectiontitle
            
        Returns:
            cleanup后's content
        """
        import re
        
        if not content:
            return content
        
        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Checkwhether tois Markdowntitle行
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()
                
                # Checkwhether tois  and sectiontitle重复's title(skipping前5行内's 重复)
                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue
                
                # 将has级别's title(#, ##, ###, #### etc.)转换粗体
                # 因sectiontitle由系统add, content不应has任何title
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # add空行
                continue
            
            # If上one行is 被skipping's title, 且当前行空, 也skipping
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        # 移除开头's 空行
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)
        
        # 移除开头's 分隔线
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            # simultaneously移除分隔线后's 空行
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def update_progress(
        cls, 
        report_id: str, 
        status: str, 
        progress: int, 
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        updatereport generationprogress
        
        前端可以throughreadprogress.jsonGet实timeprogress
        """
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Getreport generationprogress"""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Get已生成's sectionlist
        
        returnhassaved's sectionfile信息
        """
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # fromfilenameparsesection索引
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        组装completereport
        
        fromsaved's sectionfile组装completereport, 并进行titlecleanup
        """
        folder = cls._get_report_folder(report_id)
        
        # Buildreport头部
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"
        
        # by 顺序readhassectionfile
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]
        
        # 后process:cleanup整 itemsreport's titlequestion
        md_content = cls._post_process_report(md_content, outline)
        
        # Savecompletereport
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"completereport已组装: {report_id}")
        return md_content
    
    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        后processreportcontent
        
        1. 移除重复's title
        2. 保留report主title(#) and sectiontitle(##), 移除its他级别's title(###, #### etc.)
        3. cleanup多余's 空行 and 分隔线
        
        Args:
            content: 原始reportcontent
            outline: reportoutline
            
        Returns:
            process后's content
        """
        import re
        
        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False
        
        # 收集outline's hassectiontitle
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Checkwhether tois title行
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Checkwhether tois 重复title(in 连续5行内出现相同content's title)
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    # Skip重复title and its后's 空行
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue
                
                # title层级process:
                # - # (level=1) only保留report主title
                # - ## (level=2) 保留sectiontitle
                # - ###  and 以下 (level>=3) 转换粗体text
                
                if level == 1:
                    if title == outline.title:
                        # 保留report主title
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # sectiontitleerroruse了#, 修正##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # its他one级title转粗体
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # 保留sectiontitle
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # 非section's 二级title转粗体
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # ###  and 以下级别's title转换粗体text
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False
                
                i += 1
                continue
            
            elif stripped == '---' and prev_was_heading:
                # Skiptitle后紧跟's 分隔线
                i += 1
                continue
            
            elif stripped == '' and prev_was_heading:
                # title后only保留one items空行
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False
            
            else:
                processed_lines.append(line)
                prev_was_heading = False
            
            i += 1
        
        # Clean up连续's 多 items空行(保留最多2 items)
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @classmethod
    def save_report(cls, report: Report) -> None:
        """Savereport元信息 and completereport"""
        cls._ensure_report_folder(report.report_id)
        
        # Save元信息JSON
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Save outline
        if report.outline:
            cls.save_outline(report.report_id, report.outline)
        
        # SavecompleteMarkdownreport
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(f"reportsaved: {report.report_id}")
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Getreport"""
        path = cls._get_report_path(report_id)
        
        if not os.path.exists(path):
            # 兼容旧format:check直接存储in reportsdirectory下's file
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 重建Reportobject
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        # Ifmarkdown_content空, attemptfromfull_report.mdread
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """based onsimulationIDGetreport"""
        cls._ensure_reports_dir()
        
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # 新format:file夹
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # 兼容旧format:JSONfile
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report
        
        return None
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """Listreport"""
        cls._ensure_reports_dir()
        
        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # 新format:file夹
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # 兼容旧format:JSONfile
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
        
        # by createtime倒序
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Deletereport(整 itemsfile夹)"""
        import shutil
        
        folder_path = cls._get_report_folder(report_id)
        
        # 新format:delete整 itemsfile夹
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"reportfile夹deleted: {report_id}")
            return True
        
        # 兼容旧format:delete单独's file
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
