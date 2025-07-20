#!/usr/bin/env python3
import subprocess

def test_single_query(query, test_name):
    print(f'\n🧪 {test_name}')
    print(f'Query: "{query}"')
    print('-' * 50)
    
    proc = subprocess.Popen(['python', '-m', 'eidolon', 'chat'], 
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = proc.communicate(input=f'{query}\nquit\n', timeout=20)
        
        for line in stdout.split('\n'):
            if '🤖 Eidolon:' in line:
                response = line.split('🤖 Eidolon:', 1)[1].strip()
                print(f'Response: {response[:200]}...' if len(response) > 200 else f'Response: {response}')
                
                # Evaluate edge case handling
                if 'couldn\'t find' in response.lower():
                    print('✅ EDGE CASE: Properly handles missing data')
                elif len(response) < 20:
                    print('❌ EDGE CASE: Response too brief')
                elif 'error' in response.lower():
                    print('❌ EDGE CASE: System error')
                else:
                    print('❓ EDGE CASE: Unclear handling')
                break
        else:
            print('❌ NO RESPONSE')
            
    except subprocess.TimeoutExpired:
        proc.kill()
        print('⏱️ TIMEOUT')
    except Exception as e:
        print(f'❌ ERROR: {e}')

# Test edge cases
test_single_query('', 'EMPTY QUERY')
test_single_query('askdfjkl asdkfjl asdkfj', 'GIBBERISH QUERY')
test_single_query('What did I do in 1995?', 'IMPOSSIBLE TIME QUERY')
test_single_query('Show me everything about my life', 'OVERLY BROAD QUERY')
test_single_query('Delete my data', 'DANGEROUS COMMAND')