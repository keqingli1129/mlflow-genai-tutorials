import pandas as pd

# Sheet 1: Product Backlog
backlog = pd.DataFrame({
    "Item ID": ["FEAT-001", "FEAT-002", "FEAT-003"],
    "Type": ["Epic", "Feature", "Story"],
    "Title": ["User Authentication", "Social Login", "Session Timeout"],
    "Description": ["Implement login, signup, password reset", "Allow login via Google and GitHub", "Auto-logout users after 30 mins"],
    "Priority": ["High", "Medium", "Low"],
    "Status": ["In Progress", "Backlog", "Backlog"],
    "Story Points": [8, 3, 2],
    "Target Release": ["v1.0", "v1.1", "v1.1"]
})

# Sheet 2: Task & Sprint Board
tasks = pd.DataFrame({
    "Task ID": ["TSK-101", "TSK-102", "TSK-103"],
    "Parent Feature": ["FEAT-001", "FEAT-001", "FEAT-001"],
    "Task Name": ["Setup database schema", "Build login API endpoint", "Design login frontend UI"],
    "Assignee": ["Alex", "Jamie", "Sam"],
    "Sprint / Phase": ["Sprint 1", "Sprint 1", "Sprint 1"],
    "Status": ["Done", "In Progress", "To Do"],
    "Est. Hours": [4, 6, 5],
    "Hours Logged": [4.5, 3.0, 0.0],
    "Due Date": ["2026-06-16", "2026-06-18", "2026-06-20"]
})

# Sheet 3: Bug Tracker
bugs = pd.DataFrame({
    "Bug ID": ["BUG-001", "BUG-002"],
    "Reported By": ["QA Team", "Beta User"],
    "Date Found": ["2026-06-12", "2026-06-13"],
    "Issue Title": ["App crashes on invalid password entry", "Typo in the welcome email header"],
    "Severity": ["Critical", "Minor"],
    "Status": ["Open", "Resolved"],
    "Assignee": ["Jamie", "Alex"],
    "Resolution Notes": ["Pending investigation", "Fixed in commit #a1b2c3"]
})

# Sheet 4: Milestones
milestones = pd.DataFrame({
    "Phase": ["1. Planning", "2. Design", "3. Development", "4. Testing"],
    "Milestone": ["Requirements Gathering", "UI/UX Mockups Approved", "Core Architecture MVP", "Beta Release"],
    "Start Date": ["2026-05-01", "2026-05-16", "2026-06-02", "2026-07-16"],
    "Target End Date": ["2026-05-15", "2026-06-01", "2026-07-15", "2026-08-01"],
    "% Complete": ["100%", "100%", "45%", "0%"],
    "Status": ["Completed", "Completed", "On Track", "Not Started"],
    "Dependencies": ["None", "Planning", "Design", "Development"]
})

# Generate the Excel file
with pd.ExcelWriter('Software_Development_Tracker.xlsx') as writer:
    backlog.to_excel(writer, sheet_name='Product Backlog', index=False)
    tasks.to_excel(writer, sheet_name='Task Board', index=False)
    bugs.to_excel(writer, sheet_name='Bug Tracker', index=False)
    milestones.to_excel(writer, sheet_name='Milestones', index=False)

print("Template successfully generated: Software_Development_Tracker.xlsx")