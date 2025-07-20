#!/usr/bin/env python3
import subprocess

def test_conversation(queries, test_name):
    print(f'\n🧪 {test_name}')
    print('-' * 60)
    
    # Create input with all queries
    input_text = '\n'.join(queries) + '\nquit\n'
    
    proc = subprocess.Popen(['python', '-m', 'eidolon', 'chat'], 
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = proc.communicate(input=input_text, timeout=30)
        
        lines = stdout.split('\n')
        responses = []
        
        for line in lines:
            if '🤖 Eidolon:' in line:
                response = line.split('🤖 Eidolon:', 1)[1].strip()
                responses.append(response)
        
        # Analyze conversation flow
        for i, (query, response) in enumerate(zip(queries, responses), 1):
            print(f'Q{i}: {query}')
            print(f'A{i}: {response[:150]}...' if len(response) > 150 else f'A{i}: {response}')
            print()
        
        # Evaluate conversation quality
        if len(responses) != len(queries):
            print('❌ CONVERSATION FLOW: Missing responses')
        elif len(responses) > 1:
            # Check if follow-ups reference previous context
            second_response = responses[1].lower() if len(responses) > 1 else ''
            if any(word in second_response for word in ['video', 'tutorial', 'claude', 'ai']):
                print('✅ CONVERSATION FLOW: Follow-up shows context awareness')
            else:
                print('❓ CONVERSATION FLOW: Limited context awareness')
        else:
            print('❓ CONVERSATION FLOW: Single response only')
            
    except subprocess.TimeoutExpired:
        proc.kill()
        print('⏱️ CONVERSATION TIMEOUT: Too slow for natural flow')
    except Exception as e:
        print(f'❌ CONVERSATION ERROR: {e}')

# Test conversation scenarios
test_conversation([
    'What was the last YouTube video I watched?',
    'When did I watch it?', 
    'Was it helpful?'
], 'YOUTUBE FOLLOW-UP CONVERSATION')

# Test practical workflow
test_conversation([
    'What did I work on today?',
    'How long did it take?',
    'Did I encounter any problems?'
], 'WORK PRODUCTIVITY CONVERSATION')