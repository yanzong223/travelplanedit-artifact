"""
LLM prompts and templates for TPE system.

Provides structured prompts for different agent operations.
"""

import json
from typing import Any, Dict, List, Optional


class PromptTemplates:
    """Collection of prompt templates for various operations."""

    # Constraint Extraction Prompts
    CONSTRAINT_EXTRACTION = """You are a travel planning constraint extraction expert. Given a user's natural language edit request and their current travel plan, extract structured constraints that need to be satisfied.

Current Travel Plan Context:
{current_plan}

User Edit Request:
{edit_request}

Extract the following types of constraints:
1. Temporal constraints (time windows, sequences, durations)
2. Budget constraints (cost limits, spending categories)
3. Geographic constraints (distances, locations, proximity)
4. Logical constraints (dependencies, exclusions, requirements)
5. User preferences (must-have vs nice-to-have)

Respond with a structured JSON object containing:
{{
  "constraints": [
    {{
      "type": "TEMPORAL|BUDGET|GEOGRAPHIC|LOGICAL|PREFERENCE",
      "description": "Human-readable description",
      "variables": ["variable_id_1", "variable_id_2"],
      "parameters": {{
        "key": "value"
      }},
      "is_hard": true/false,
      "weight": 0.0-1.0
    }}
  ],
  "missing_information": [
    {{
      "type": "what_is_missing",
      "description": "what information is needed",
      "impact": "how this affects the edit"
    }}
  ]
}}"""

    # Conflict Classification Prompts
    CONFLICT_CLASSIFICATION = """You are a travel planning conflict analysis expert. Given violated constraints and proposed changes, classify conflicts as either missing information or logical contradictions.

Violated Constraints:
{violated_constraints}

Proposed Changes:
{proposed_changes}

Current Plan Context:
{current_plan}

For each conflict, classify as:
1. "MISSING_INFO" - Need more information from user or external sources
2. "LOGICAL_CONTRADICTION" - Cannot be satisfied with any reasonable information
3. "PREFERENCE_CONFLICT" - User preferences conflict with constraints

Respond with JSON:
{{
  "conflicts": [
    {{
      "constraint_id": "constraint_id",
      "conflict_type": "MISSING_INFO|LOGICAL_CONTRADICTION|PREFERENCE_CONFLICT",
      "core_type": "fundamental conflict category",
      "gaps": [
        "specific information needed to resolve",
        "another gap description"
      ],
      "description": "human-readable explanation",
      "severity": "HIGH|MEDIUM|LOW",
      "resolvable": true/false
    }}
  ],
  "overall_assessment": "summary of all conflicts",
  "recommended_actions": [
    "action 1 to resolve conflicts",
    "action 2 to resolve conflicts"
  ]
}}"""

    # TODO 检验当前conflict classification的prompt是否合理
    # Conflict Classification V2 (Standardized)
    CONFLICT_CLASSIFICATION_V2 = """You are a travel planning conflict analysis expert. Given a current travel plan and new constraints, classify why conflicts exist.

Current Travel Plan:
{plan_summary}

New Constraints to Apply:
{constraints_summary}

Analyze the conflicts and determine the conflict type:

**MISSING_INFO** - Choose this when we need to retrieve external data:
- Need to search for alternative hotels/flights/activities from databases
- Need to find cheaper/better options that don't exist in current plan
- Need to add new items (museums, restaurants, attractions) not in the plan
- Budget constraints require finding new lower-priced options
- User wants to replace existing items with alternatives from external sources
- Need to search for options meeting specific criteria (price, location, time)

Examples of MISSING_INFO:
- "Reduce hotel budget to 200 yuan" → Need to search for cheaper hotels
- "Add a museum visit" → Need to search for available museums
- "Find cheaper flights" → Need to search flight database
- "Replace hotel with better location" → Need to search for alternatives

**LOGICAL_ISSUE** - Choose this when no external data needed:
- Can be resolved by adjusting times/dates of existing items
- Can be resolved by reordering or rescheduling existing activities
- Issue is with internal plan structure or timing conflicts
- Mathematical/logical impossibilities (conflicting times, etc.)
- User constraints are self-contradictory

Examples of LOGICAL_ISSUE:
- "Change activity order" → Just reorder existing items
- "Adjust departure time by 1 hour" → Modify existing flight
- "Swap day 1 and day 2 activities" → Rearrange plan

**IMPORTANT**: When in doubt, if satisfying the constraint might benefit from having more options or alternatives, choose MISSING_INFO.

If MISSING_INFO, provide detailed retrieval instructions with ALL relevant parameters from the constraints.

Respond with JSON:
{{
  "conflict_reason": "Detailed explanation of why conflicts exist",
  "conflict_type": "LOGICAL_ISSUE or MISSING_INFO",
  "reasoning": "Brief summary of analysis logic",
  "retrieval_instructions": [
    {{
      "retrieval_type": "search_flights|search_hotels|search_activities|search_restaurants|search_attractions",
      "parameters": {{
        // CRITICAL: Extract ALL relevant parameters from the constraints!
        // For flights:
        "origin": "departure city",
        "destination": "arrival city", 
        "departure_time": "datetime or time range",
        "arrival_time": "datetime or time range",
        // For hotels:
        "location": "city or poi name",
        "check_in": "date",
        "check_out": "date",
        "max_price": "number",
        // For activities/restaurants/attractions:
        "location": "city or poi name (e.g. '中国水利博物馆')",
        "earliest_time": "HH:MM format (e.g. '10:30')",
        "latest_time": "HH:MM format (e.g. '13:00')",
        "day_index": "day number in trip",
        "max_distance_km": "number",
        "city": "city name in Chinese",
        // For budget constraints:
        "max_budget": "number",
        "min_budget": "number"
      }},
      "priority": 1-10,
      "rationale": "Clear explanation of why this retrieval is needed"
    }}
  ]
}}

**CRITICAL REQUIREMENTS**:
1. **retrieval_type** must be exact: search_flights, search_hotels, search_activities, search_restaurants, or search_attractions
2. **parameters** must include ALL relevant constraints (time, location, budget, etc.) from the constraint summary
3. **priority** should be 8-10 for critical retrievals, 5-7 for important, 1-4 for optional
4. **rationale** must explain what data is needed and why
5. Extract actual values from constraints - don't use placeholders!

Note: Only include retrieval_instructions if conflict_type is MISSING_INFO.
"""

    # Edit Candidate Generation Prompts
    EDIT_CANDIDATE_GENERATION = """You are a travel planning editor expert. Generate up to {max_candidates} different edit candidates to satisfy the user's request while maintaining all constraints.

User Request:
{edit_request}

Current Plan:
{current_plan}

Constraints to Satisfy:
{constraints}

Allowed Operations: {allowed_operations}

For each candidate, generate a sequence of edit operations. Each operation must be one of:
- Insert: Add new activity/flight/hotel
- Replace: Replace existing element with alternative
- AdjustTime: Change time/duration of existing element
- ModeChange: Change transportation mode or accommodation type

Respond with JSON:
{{
  "candidates": [
    {{
      "id": "candidate_1",
      "description": "brief description of this approach",
      "operations": [
        {{
          "type": "Insert|Replace|AdjustTime|ModeChange",
          "target": "variable_id_or_new_element",
          "changes": {{
            "field": "new_value",
            "another_field": "new_value"
          }},
          "reasoning": "why this operation helps"
        }}
      ],
      "estimated_impact": {{
        "fields_changed": 3,
        "constraints_affected": 2,
        "user_satisfaction": "HIGH|MEDIUM|LOW"
      }},
      "confidence": 0.0-1.0
    }}
  ],
  "selected_candidate": "candidate_1",
  "reasoning": "why this candidate was selected as best",
  "alternative_considerations": "brief notes on rejected candidates"
}}"""

    # Natural Language Explanation Prompts
    EXPLANATION_GENERATION = """You are a travel planning assistant. Generate a clear, natural language explanation for the automated decision made.

Operation Performed:
{operation}

Result:
{result}

User Request:
{original_request}

Generate an explanation that:
1. Acknowledges the user's original request
2. Explains what was changed and why
3. Mentions any constraints that influenced the decision
4. Provides confidence level
5. Offers next steps or alternatives if needed

Keep it concise (2-3 sentences) and user-friendly. Avoid technical jargon.

Explanation:"""

    # Flight Search Prompts
    FLIGHT_SEARCH = """Find flights matching these criteria:

Origin: {origin}
Destination: {destination}
Departure Time: {departure_time}
Constraints: {constraints}

Return up to 5 options with:
- Flight number and airline
- Departure/arrival times
- Price
- Available seats
- Any relevant restrictions

Format as JSON array of flight objects."""

    # Hotel Search Prompts
    HOTEL_SEARCH = """Find hotels matching these criteria:

Location: {location}
Check-in: {check_in}
Check-out: {check_out}
Constraints: {constraints}

Return up to 5 options with:
- Hotel name and rating
- Room type and occupancy
- Price per night
- Available rooms
- Amenities
- Cancellation policy

Format as JSON array of hotel objects."""

    # Activity Search Prompts
    ACTIVITY_SEARCH = """Find activities/attractions matching these criteria:

Location: {location}
Date: {date}
Constraints: {constraints}

Return up to 5 options with:
- Activity name and type
- Duration
- Price
- Operating hours
- Location/address
- Booking requirements
- Description

Format as JSON array of activity objects."""


class PromptBuilder:
    """Builds prompts with context and parameters."""

    @staticmethod
    def build_constraint_extraction_prompt(edit_request: str, current_plan: str) -> str:
        """Build constraint extraction prompt."""
        return PromptTemplates.CONSTRAINT_EXTRACTION.format(
            current_plan=current_plan, edit_request=edit_request
        )

    @staticmethod
    def build_conflict_classification_prompt(
        violated_constraints: List[str],
        proposed_changes: Dict[str, Any],
        current_plan: str,
    ) -> str:
        """Build conflict classification prompt (legacy)."""
        return PromptTemplates.CONFLICT_CLASSIFICATION.format(
            violated_constraints="\n".join(violated_constraints),
            proposed_changes=json.dumps(proposed_changes, indent=2),
            current_plan=current_plan,
        )

    @staticmethod
    def build_conflict_classification_prompt_v2(
        plan_summary: str, constraints_summary: str
    ) -> str:
        """Build conflict classification prompt (standardized)."""
        return PromptTemplates.CONFLICT_CLASSIFICATION_V2.format(
            plan_summary=plan_summary, constraints_summary=constraints_summary
        )

    @staticmethod
    def build_edit_candidate_prompt(
        edit_request: str,
        current_plan: str,
        constraints: List[str],
        max_candidates: int = 5,
        allowed_operations: Optional[List[str]] = None,
    ) -> str:
        """Build edit candidate generation prompt."""
        ops = allowed_operations or ["Insert", "Replace", "AdjustTime", "ModeChange"]
        return PromptTemplates.EDIT_CANDIDATE_GENERATION.format(
            max_candidates=max_candidates,
            edit_request=edit_request,
            current_plan=current_plan,
            constraints="\n".join(constraints),
            allowed_operations=", ".join(ops),
        )

    @staticmethod
    def build_explanation_prompt(
        operation: str, result: str, original_request: str
    ) -> str:
        """Build explanation generation prompt."""
        return PromptTemplates.EXPLANATION_GENERATION.format(
            operation=operation, result=result, original_request=original_request
        )

    @staticmethod
    def build_flight_search_prompt(
        origin: str,
        destination: str,
        departure_time: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build flight search prompt."""
        return PromptTemplates.FLIGHT_SEARCH.format(
            origin=origin,
            destination=destination,
            departure_time=departure_time,
            constraints=json.dumps(constraints or {}, indent=2),
        )

    @staticmethod
    def build_hotel_search_prompt(
        location: str,
        check_in: str,
        check_out: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build hotel search prompt."""
        return PromptTemplates.HOTEL_SEARCH.format(
            location=location,
            check_in=check_in,
            check_out=check_out,
            constraints=json.dumps(constraints or {}, indent=2),
        )

    @staticmethod
    def build_activity_search_prompt(
        location: str, date: str, constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build activity search prompt."""
        return PromptTemplates.ACTIVITY_SEARCH.format(
            location=location,
            date=date,
            constraints=json.dumps(constraints or {}, indent=2),
        )
