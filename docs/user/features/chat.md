# Chat Interface

Eidolon's chat interface provides a natural language way to interact with your digital memory. Ask questions, request analysis, and get insights using conversational AI powered by your captured activity.

## 🎯 Getting Started

### Launch Chat Mode

```bash
# Start interactive chat session
python -m eidolon chat

# Start with specific context
python -m eidolon chat --context "work"

# Web-based chat interface
python -m eidolon serve --chat-only
```

### Basic Interaction

Once in chat mode, you can ask questions naturally:

```
> What did I work on today?
> Show me all emails from last week
> Find that Python article I read yesterday
> Summarize my meeting notes from Tuesday
> What websites did I visit about machine learning?
```

## 💬 Chat Commands

### Question Types

#### **Activity Queries**
- "What did I do this morning?"
- "Show me my activity timeline for yesterday"
- "What applications did I use most today?"
- "How much time did I spend coding this week?"

#### **Content Search**
- "Find emails containing 'project deadline'"
- "Show me all documents about the quarterly report"
- "Where did I see information about API documentation?"
- "Find that GitHub repository I was looking at"

#### **Analysis Requests**
- "Analyze my productivity patterns this month"
- "What are my most frequent distractions?"
- "Compare my focus time this week vs last week"
- "Identify my peak productivity hours"

#### **Context Retrieval**
- "What was I working on before the meeting?"
- "Get context for my conversation with Sarah"
- "Show me related content about the new project"
- "Find everything related to client XYZ"

### Special Commands

```bash
# Within chat mode:
/help          # Show available commands
/status        # System status
/stats         # Quick statistics
/export        # Export conversation
/clear         # Clear chat history
/config        # Show current settings
/quit          # Exit chat mode
```

## 🔧 Configuration

### Chat Preferences

```bash
# Set default AI model for chat
python -m eidolon config set chat.default_model "claude-3"

# Configure response length
python -m eidolon config set chat.response_length "detailed"  # brief, normal, detailed

# Set conversation memory
python -m eidolon config set chat.memory_turns 10
```

### Advanced Settings

```yaml
# ~/.eidolon/config/chat.yaml
chat:
  default_model: "claude-3"
  response_length: "detailed"
  memory_turns: 10
  
  # Response preferences
  include_sources: true
  show_confidence: true
  suggest_followups: true
  
  # Context settings
  max_context_items: 20
  relevance_threshold: 0.7
  time_decay_factor: 0.9
  
  # UI preferences
  streaming_responses: true
  syntax_highlighting: true
  auto_save_conversations: true
```

## 🤖 AI Models

### Available Models

#### **Local Models** (Fast, Private)
- **Basic Chat**: Quick responses for simple queries
- **Retrieval Model**: Optimized for finding specific information
- **Analysis Model**: Local analysis and summarization

#### **Cloud Models** (Powerful, Feature-Rich)
- **Claude-3**: Advanced reasoning and analysis
- **GPT-4**: General-purpose conversational AI
- **Gemini**: Google's multimodal AI with vision capabilities

### Model Selection

```bash
# Use specific model for session
python -m eidolon chat --model "claude-3"

# Switch models within chat
> /model gpt-4
> /model local-basic

# Configure model preferences
python -m eidolon config set chat.preferred_models "['claude-3', 'gpt-4', 'local-basic']"
```

## 🎨 Chat Features

### Conversation Context

Eidolon maintains context throughout your conversation:

```
> What emails did I get today?
[Eidolon shows email list]

> Summarize the important ones
[Eidolon understands "ones" refers to the emails from previous question]

> Who sent the most messages?
[Continues context from email discussion]
```

### Rich Responses

Responses can include:
- **Text summaries** with key insights
- **Structured data** in tables or lists
- **Visual timelines** of activity
- **Links to source content** for verification
- **Suggested follow-up questions**

### Multi-turn Conversations

```
User: Find documents about the Q4 planning
Eidolon: I found 15 documents related to Q4 planning from the last month...

User: Which ones are most recent?
Eidolon: The 3 most recent Q4 planning documents are...

User: Open the presentation from Sarah
Eidolon: Opening "Q4 Strategy Presentation - Sarah.pptx" captured on...
```

## 🔍 Advanced Queries

### Time-based Queries

```
> What did I work on between 2 PM and 4 PM yesterday?
> Show me my activity from last Monday
> Find emails from this morning before 10 AM
> What websites did I visit during lunch break?
```

### Filtered Searches

```
> Find all Python code I wrote this week
> Show me only work-related emails from today
> Find meeting notes that mention "budget"
> Show me documents I created, not just viewed
```

### Analytical Queries

```
> What patterns do you see in my work habits?
> Am I more productive in the morning or afternoon?
> How has my focus time changed over the past month?
> What applications distract me the most?
```

### Comparative Analysis

```
> Compare my productivity this week vs last week
> How much more time did I spend on email than coding?
> Show me the difference in my activity patterns Mon vs Fri
> Compare my focus during meetings vs individual work
```

## 🎛️ Chat Modes

### Standard Mode (Default)
- Natural conversation flow
- Full context awareness
- Detailed responses with sources

### Quick Mode
```bash
python -m eidolon chat --mode quick
```
- Brief, direct answers
- Faster response times
- Minimal context loading

### Analysis Mode
```bash
python -m eidolon chat --mode analysis
```
- Deep analytical responses
- Statistical insights
- Pattern recognition focus

### Search Mode
```bash
python -m eidolon chat --mode search
```
- Optimized for finding specific information
- Precise result matching
- Minimal interpretation, maximum accuracy

## 📊 Integration Features

### Export Conversations

```bash
# Export current conversation
> /export conversation.md

# Export with metadata
> /export detailed_conversation.json

# Auto-export all conversations
python -m eidolon config set chat.auto_export true
```

### Share Insights

```bash
# Generate shareable summary
> Create a summary I can share with my team about yesterday's progress

# Export specific findings
> Export the productivity analysis as a report

# Create presentation content
> Turn this week's insights into bullet points for my status update
```

### Integration with Other Tools

```bash
# Send to calendar
> Add a reminder about the follow-up mentioned in today's emails

# Create tasks
> Create a task list from today's meeting notes

# Export to notes app
> Save this analysis to my notes about project planning
```

## 🔒 Privacy and Security

### Data Handling

- **Local Processing**: Basic queries processed locally when possible
- **Selective Cloud**: Only send necessary context to cloud models
- **No Personal Info**: Automatically redact sensitive information
- **User Control**: Choose what information to include in cloud requests

### Privacy Controls

```bash
# Enable local-only mode
python -m eidolon config set chat.local_only true

# Configure data sharing preferences
python -m eidolon config set chat.cloud_data_policy "minimal"

# Set redaction patterns
python -m eidolon config add chat.redact_patterns "credit card"
```

## 🚨 Troubleshooting

### Common Issues

#### Chat Not Responding
```bash
# Check system status
python -m eidolon status

# Restart chat service
python -m eidolon restart chat

# Clear chat cache
python -m eidolon clear-cache chat
```

#### Poor Response Quality
```bash
# Rebuild search index
python -m eidolon reindex

# Update AI models
python -m eidolon update models

# Check data availability
python -m eidolon stats data
```

#### Slow Responses
```bash
# Switch to local model
> /model local-basic

# Enable quick mode
python -m eidolon chat --mode quick

# Reduce context size
python -m eidolon config set chat.max_context_items 10
```

## 🎯 Best Practices

### Effective Questioning

1. **Be Specific**: "Show me Python files I edited today" vs "What did I do?"
2. **Use Time Ranges**: "emails from this morning" vs "recent emails"
3. **Provide Context**: "In the meeting about project X, what was decided?"
4. **Follow Up**: Build on previous questions for deeper insights

### Conversation Management

1. **Start Fresh**: Use `/clear` for new topics
2. **Export Important**: Save valuable insights with `/export`
3. **Review Context**: Check what information is being used
4. **Switch Models**: Use appropriate model for the task

### Privacy Awareness

1. **Review Responses**: Check that sensitive info isn't exposed
2. **Use Local Mode**: For sensitive queries, stay local
3. **Regular Cleanup**: Clear conversation history periodically
4. **Configure Redaction**: Set up automatic sensitive data filtering

---

The chat interface transforms your captured digital activity into an intelligent, conversational assistant that knows your work patterns, preferences, and context.