import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# ==================== CONFIGURATION ====================

st.set_page_config(
    page_title="Automated Task Scheduler",
    layout="wide",
    initial_sidebar_state="expanded"
)

POMODORO_WORK = 25
POMODORO_SHORT_BREAK = 5
POMODORO_LONG_BREAK = 15
POMODOROS_BEFORE_LONG_BREAK = 4

TAG_COLORS = {
    "Work": "#FF6B6B",
    "Exercise": "#4ECDC4",
    "Learning": "#FFE66D",
    "Personal": "#95E1D3",
    "Chores": "#F38181",
    "Break": "#AA96DA",
    "Meals": "#FCBAD3",
    "Other": "#A8D8EA",
    "Pomodoro Break": "#FFD23F"
}

# ==================== DATA STRUCTURES ====================

@dataclass
class TimeSlot:
    task_name: str
    tag: str
    start_time: datetime
    end_time: datetime
    is_fixed: bool
    is_break: bool = False
    priority: int = 0
    pomodoro_number: Optional[int] = None
    parent_task_id: Optional[str] = None

    @property
    def duration_minutes(self) -> int:
        return int((self.end_time - self.start_time).total_seconds() / 60)

@dataclass
class AvailableGap:
    start: datetime
    end: datetime

    @property
    def duration_minutes(self) -> int:
        return int((self.end - self.start).total_seconds() / 60)

# ==================== SESSION STATE ====================

def init_session_state():
    defaults = {
        "flexible_tasks": [],
        "fixed_slots": [],
        "scheduled_slots": [],
        "day_start_hour": 6,
        "day_end_hour": 23,
        "task_counter": 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==================== CORE LOGIC ====================

def find_all_gaps(fixed_slots: List[TimeSlot], day_start: datetime, day_end: datetime) -> List[AvailableGap]:
    if not fixed_slots:
        return [AvailableGap(day_start, day_end)]
    gaps = []
    sorted_fixed = sorted(fixed_slots, key=lambda x: x.start_time)
    if sorted_fixed[0].start_time > day_start:
        gaps.append(AvailableGap(day_start, sorted_fixed[0].start_time))
    for i in range(len(sorted_fixed) - 1):
        gap_start = sorted_fixed[i].end_time
        gap_end = sorted_fixed[i + 1].start_time
        if gap_start < gap_end:
            gaps.append(AvailableGap(gap_start, gap_end))
    if sorted_fixed[-1].end_time < day_end:
        gaps.append(AvailableGap(sorted_fixed[-1].end_time, day_end))
    return gaps

def calculate_pomodoro_segments(total_work_minutes: int) -> List[Tuple[int, int]]:
    segments = []
    remaining = total_work_minutes
    pomodoro_count = 0
    while remaining > 0:
        work_time = min(POMODORO_WORK, remaining)
        remaining -= work_time
        if remaining > 0:
            pomodoro_count += 1
            break_time = (POMODORO_LONG_BREAK if pomodoro_count % POMODOROS_BEFORE_LONG_BREAK == 0 else POMODORO_SHORT_BREAK)
        else:
            break_time = 0
        segments.append((work_time, break_time))
    return segments

def fit_task_in_gaps(task_name: str, tag: str, work_minutes: int, priority: int,
                     gaps: List[AvailableGap], task_id: str) -> List[TimeSlot]:
    slots = []
    remaining_work = work_minutes
    pomodoro_segments = calculate_pomodoro_segments(work_minutes)
    segment_index = 0
    pomodoro_count = 0

    for gap in gaps:
        if remaining_work <= 0:
            break
        current_time = gap.start
        while remaining_work > 0 and current_time < gap.end:
            if segment_index >= len(pomodoro_segments):
                break
            work_duration, break_duration = pomodoro_segments[segment_index]
            remaining_gap = int((gap.end - current_time).total_seconds() / 60)
            if remaining_gap >= work_duration:
                work_end = current_time + timedelta(minutes=work_duration)
                pomodoro_count += 1
                slots.append(TimeSlot(
                    task_name=task_name, tag=tag, start_time=current_time, end_time=work_end,
                    is_fixed=False, is_break=False, priority=priority,
                    pomodoro_number=pomodoro_count, parent_task_id=task_id
                ))
                remaining_work -= work_duration
                current_time = work_end
                if break_duration > 0 and remaining_work > 0:
                    remaining_gap_after_work = int((gap.end - current_time).total_seconds() / 60)
                    if remaining_gap_after_work >= break_duration:
                        break_end = current_time + timedelta(minutes=break_duration)
                        slots.append(TimeSlot(
                            task_name=f"{task_name} - Break", tag="Pomodoro Break",
                            start_time=current_time, end_time=break_end,
                            is_fixed=False, is_break=True, priority=priority, parent_task_id=task_id
                        ))
                        current_time = break_end
                segment_index += 1
            else:
                break
    return slots

def update_gaps_after_scheduling(gaps: List[AvailableGap], scheduled_slots: List[TimeSlot]) -> List[AvailableGap]:
    if not scheduled_slots:
        return gaps
    new_gaps = []
    for gap in gaps:
        remaining_gap_start = gap.start
        overlapping = sorted([s for s in scheduled_slots 
                            if s.start_time < gap.end and s.end_time > gap.start],
                           key=lambda x: x.start_time)
        for slot in overlapping:
            if remaining_gap_start < slot.start_time:
                new_gaps.append(AvailableGap(remaining_gap_start, slot.start_time))
            remaining_gap_start = max(remaining_gap_start, slot.end_time)
        if remaining_gap_start < gap.end:
            new_gaps.append(AvailableGap(remaining_gap_start, gap.end))
    return new_gaps

def get_last_scheduled_tag(schedule: List[TimeSlot]) -> Optional[str]:
    """Get the tag of the last non-break flexible task in schedule"""
    for slot in reversed(schedule):
        if not slot.is_fixed and not slot.is_break:
            return slot.tag
    return None

def can_schedule_task_here(task_tag: str, schedule: List[TimeSlot], start_time: datetime) -> bool:
    """
    Check if task with given tag can be scheduled at start_time.
    Rule: Two tasks with same tag cannot be consecutive (unless both are fixed)
    """
    # Find the slot right before start_time
    slots_before = [s for s in schedule if s.end_time <= start_time and not s.is_break]

    if not slots_before:
        return True  # No previous task, can schedule

    # Get the last task before this position
    last_task = max(slots_before, key=lambda x: x.end_time)

    # Allow consecutive fixed tasks with same tag
    if last_task.is_fixed:
        return True

    # Check if last task has same tag as this one
    if last_task.tag == task_tag:
        return False  # Cannot schedule same tag consecutively

    return True

def build_complete_schedule(flexible_tasks: List[Dict], fixed_slots: List[TimeSlot],
                           day_start: datetime, day_end: datetime,
                           insertion_task: Optional[Dict] = None,
                           insertion_time: Optional[datetime] = None) -> List[TimeSlot]:
    schedule = fixed_slots.copy()
    gaps = find_all_gaps(fixed_slots, day_start, day_end)

    # Sort by priority first
    sorted_tasks = sorted(flexible_tasks, key=lambda x: x["priority"])

    if insertion_task and insertion_time:
        in_fixed = any(slot.start_time <= insertion_time < slot.end_time for slot in fixed_slots)
        if in_fixed:
            blocking_fixed = [s for s in fixed_slots if s.start_time <= insertion_time < s.end_time][0]
            gaps = [g for g in gaps if g.start >= blocking_fixed.end_time]
        else:
            conflicting = [s for s in schedule if not s.is_fixed and s.start_time <= insertion_time < s.end_time]
            if conflicting:
                parent_id = conflicting[0].parent_task_id
                schedule = [s for s in schedule if s.parent_task_id != parent_id]
            gaps = [AvailableGap(max(g.start, insertion_time), g.end) for g in gaps if g.end > insertion_time]

    # Schedule tasks with tag-consecutiveness check
    scheduled_count = 0
    max_attempts = len(sorted_tasks) * 2  # Prevent infinite loops

    while scheduled_count < len(sorted_tasks) and max_attempts > 0:
        max_attempts -= 1
        scheduled_any = False

        for task in sorted_tasks:
            if task.get("scheduled"):
                continue

            task_id = task.get("id", f"task_{st.session_state.task_counter}")
            task["id"] = task_id

            # Try to find gaps where this task can be scheduled
            for gap in gaps:
                # Check if we can schedule this task at the start of this gap
                if can_schedule_task_here(task["tag"], schedule, gap.start):
                    # Try to fit task in this gap
                    task_slots = fit_task_in_gaps(
                        task["name"], task["tag"], task["duration"],
                        task["priority"], [gap], task_id
                    )

                    if task_slots:  # Successfully scheduled
                        schedule.extend(task_slots)
                        gaps = update_gaps_after_scheduling(gaps, task_slots)
                        task["scheduled"] = True
                        scheduled_count += 1
                        scheduled_any = True
                        break

            if task.get("scheduled"):
                break

        if not scheduled_any:
            # If no tasks could be scheduled in this iteration, schedule remaining tasks anyway
            for task in sorted_tasks:
                if not task.get("scheduled"):
                    task_id = task.get("id", f"task_{st.session_state.task_counter}")
                    task["id"] = task_id
                    task_slots = fit_task_in_gaps(
                        task["name"], task["tag"], task["duration"],
                        task["priority"], gaps, task_id
                    )
                    schedule.extend(task_slots)
                    gaps = update_gaps_after_scheduling(gaps, task_slots)
                    task["scheduled"] = True
                    scheduled_count += 1
            break

    # Clean up scheduled flags
    for task in sorted_tasks:
        task.pop("scheduled", None)

    schedule.sort(key=lambda x: x.start_time)
    return schedule

def add_flexible_task(task_name: str, duration: int, priority: int, tag: str,
                     insert_at: Optional[datetime] = None) -> Tuple[bool, str]:
    st.session_state.task_counter += 1
    task_id = f"task_{st.session_state.task_counter}"
    new_task = {"id": task_id, "name": task_name, "duration": duration, "priority": priority, "tag": tag}
    st.session_state.flexible_tasks.append(new_task)
    day_start = datetime.strptime(f"{st.session_state.day_start_hour:02d}:00", "%H:%M")
    day_end = datetime.strptime(f"{st.session_state.day_end_hour:02d}:59", "%H:%M")
    st.session_state.scheduled_slots = build_complete_schedule(
        st.session_state.flexible_tasks, st.session_state.fixed_slots, day_start, day_end,
        new_task if insert_at else None, insert_at
    )
    return True, "✅ Scheduled"

def add_fixed_task(task_name: str, tag: str, start_time: datetime, end_time: datetime) -> Tuple[bool, str]:
    if end_time <= start_time:
        return False, "End must be after start"
    new_slot = TimeSlot(task_name=task_name, tag=tag, start_time=start_time, end_time=end_time, is_fixed=True)
    st.session_state.fixed_slots.append(new_slot)
    day_start = datetime.strptime(f"{st.session_state.day_start_hour:02d}:00", "%H:%M")
    day_end = datetime.strptime(f"{st.session_state.day_end_hour:02d}:59", "%H:%M")
    st.session_state.scheduled_slots = build_complete_schedule(
        st.session_state.flexible_tasks, st.session_state.fixed_slots, day_start, day_end
    )
    return True, "✅ Added"

# ==================== UI ====================

def render_header():
    st.title("📅 Automated Task Scheduler")

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.day_start_hour = st.number_input("Start", 0, 23, st.session_state.day_start_hour)
        with col2:
            st.session_state.day_end_hour = st.number_input("End", 0, 23, st.session_state.day_end_hour)

        st.divider()
        st.header("🎯 Flexible Task")
        with st.form("flex", clear_on_submit=True):
            name = st.text_input("Task Name", placeholder="Study for exam")
            col1, col2 = st.columns(2)
            with col1:
                dur = st.number_input("Duration (min)", 5, 480, 50, step=5)
            with col2:
                pri = st.number_input("Priority", 1, 10, 3)
            tag = st.selectbox("Category", [k for k in TAG_COLORS.keys() if k != "Pomodoro Break"])
            insert = st.time_input("Insert at (optional)", value=None)
            if st.form_submit_button("➕ Add Task", use_container_width=True, type="primary") and name.strip():
                ins_time = datetime.strptime(insert.strftime("%H:%M"), "%H:%M") if insert else None
                success, msg = add_flexible_task(name, dur, pri, tag, ins_time)
                if success:
                    st.success(msg)
                    st.rerun()

        st.divider()
        st.header("📌 Fixed Task")
        with st.form("fixed", clear_on_submit=True):
            fname = st.text_input("Task Name", placeholder="Class/Meeting")
            ftag = st.selectbox("Category", list(TAG_COLORS.keys()), key="ft")
            col1, col2 = st.columns(2)
            with col1:
                fstart = st.time_input("Start", value=datetime.strptime("09:00", "%H:%M").time())
            with col2:
                fend = st.time_input("End", value=datetime.strptime("10:00", "%H:%M").time())
            if st.form_submit_button("📌 Add Fixed", use_container_width=True) and fname.strip():
                start_dt = datetime.strptime(fstart.strftime("%H:%M"), "%H:%M")
                end_dt = datetime.strptime(fend.strftime("%H:%M"), "%H:%M")
                success, msg = add_fixed_task(fname, ftag, start_dt, end_dt)
                if success:
                    st.success(msg)
                    st.rerun()

        st.divider()
        if st.button("🗑️ Clear All", use_container_width=True, type="secondary"):
            st.session_state.flexible_tasks = []
            st.session_state.fixed_slots = []
            st.session_state.scheduled_slots = []
            st.session_state.task_counter = 0
            st.rerun()

def render_stats(slots: List[TimeSlot]):
    if not slots:
        return
    total = sum(s.duration_minutes for s in slots)
    work = sum(s.duration_minutes for s in slots if not s.is_break and not s.is_fixed)
    breaks = sum(s.duration_minutes for s in slots if s.is_break)
    fixed = sum(s.duration_minutes for s in slots if s.is_fixed)
    day_start = datetime.strptime(f"{st.session_state.day_start_hour:02d}:00", "%H:%M")
    day_end = datetime.strptime(f"{st.session_state.day_end_hour:02d}:59", "%H:%M")
    total_available = int((day_end - day_start).total_seconds() / 60)
    util = (total / total_available * 100) if total_available > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Time", f"{total}m")
    with col2:
        st.metric("Work", f"{work}m")
    with col3:
        st.metric("Breaks", f"{breaks}m")
    with col4:
        st.metric("Fixed", f"{fixed}m")
    with col5:
        st.metric("Efficiency", f"{util:.0f}%")

def render_timeline(slots: List[TimeSlot]):
    if not slots:
        st.info("👈 Add tasks from sidebar to build your schedule")
        return

    fig = go.Figure()
    sorted_slots = sorted(slots, key=lambda x: x.start_time)

    for slot in sorted_slots:
        color = TAG_COLORS.get(slot.tag, "#A8D8EA")
        if slot.is_fixed:
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            r, g, b = int(r * 0.8), int(g * 0.8), int(b * 0.8)
            color = f"#{r:02x}{g:02x}{b:02x}"

        opacity = 0.6 if slot.is_break else 1.0
        pomo = f"#{slot.pomodoro_number}" if slot.pomodoro_number else ""
        typ = "Fixed" if slot.is_fixed else ("Break" if slot.is_break else "Work")

        hover = (f"<b>{slot.task_name}</b><br>"
                f"{slot.start_time.strftime('%H:%M')} - {slot.end_time.strftime('%H:%M')}<br>"
                f"{slot.duration_minutes}m • {typ}" + (f"<br>Pomodoro {pomo}" if pomo else ""))

        fig.add_trace(go.Bar(
            x=[slot.duration_minutes], y=["Schedule"], orientation='h',
            marker=dict(color=color, opacity=opacity, line=dict(color='white', width=2)),
            hovertemplate=hover + "<extra></extra>", showlegend=False,
            text=slot.task_name if not slot.is_break else "",
            textposition='inside', textfont=dict(color='white', size=11)
        ))

    fig.update_layout(
        barmode='stack', height=200, margin=dict(l=0, r=0, t=40, b=60),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="", showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode='closest',
        title={'text': '🕐 Your Day Timeline', 'font': {'size': 20}, 'x': 0.5, 'xanchor': 'center'}
    )

    st.plotly_chart(fig, use_container_width=True)

    # Time labels
    if sorted_slots:
        cols = st.columns(len(sorted_slots))
        for i, slot in enumerate(sorted_slots):
            with cols[i]:
                st.caption(slot.start_time.strftime('%H:%M'))

def render_table(slots: List[TimeSlot]):
    if not slots:
        return

    data = []
    for slot in slots:
        icons = {"fixed": "📌", "flexible": "🎯", "break": "⏸️"}
        typ = "fixed" if slot.is_fixed else ("break" if slot.is_break else "flexible")
        pomo = f" #{slot.pomodoro_number}" if slot.pomodoro_number else ""
        color = TAG_COLORS.get(slot.tag, "#A8D8EA")

        data.append({
            "Color": color,
            "⏰ Time": f"{slot.start_time.strftime('%H:%M')}-{slot.end_time.strftime('%H:%M')}",
            "📋 Task": f"{slot.task_name}{pomo}",
            "⏱️ Duration": f"{slot.duration_minutes}m",
            "🏷️ Type": f"{icons[typ]} {typ.title()}",
            "🎯 Priority": slot.priority if not slot.is_fixed else "-",
            "📁 Category": slot.tag
        })

    df = pd.DataFrame(data)

    # Display with color column
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True
    )

def main():
    init_session_state()
    render_header()
    render_sidebar()

    st.subheader("📊 Schedule Overview")
    render_stats(st.session_state.scheduled_slots)

    st.divider()

    render_timeline(st.session_state.scheduled_slots)

    st.divider()

    with st.expander("📋 Detailed Schedule", expanded=False):
        render_table(st.session_state.scheduled_slots)

    if st.session_state.fixed_slots:
        with st.expander("📊 Gap Analysis"):
            day_start = datetime.strptime(f"{st.session_state.day_start_hour:02d}:00", "%H:%M")
            day_end = datetime.strptime(f"{st.session_state.day_end_hour:02d}:59", "%H:%M")
            gaps = find_all_gaps(st.session_state.fixed_slots, day_start, day_end)
            st.write("**Available Gaps:**")
            for i, gap in enumerate(gaps, 1):
                st.write(f"{i}. {gap.start.strftime('%H:%M')}-{gap.end.strftime('%H:%M')} ({gap.duration_minutes}m)")

if __name__ == "__main__":
    main()
