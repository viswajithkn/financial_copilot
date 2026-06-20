import json
from route import route_question
from llmRouter import sharedRouter as router

def run_interactive_test():
    # Initialize the router once
    
    print("\n==============================================")
    print("🏦 Financial Analytics System - Interactive Test")
    print("==============================================")
    print("Type your financial questions below.")
    print("Type 'exit' or 'quit' to stop.")
    print("----------------------------------------------")

    while True:
        # 1. Get the question from you
        question = input("\n💬 Ask a question: ").strip()
        
        if not question:
            continue
            
        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        print("\n🧠 Step 1: Thinking...")
        print("\n⚙️ Step 2: Running tools against databases...")
        
        # 3. Test the full execution against your SQLite and Vector DBs
        try:
            results = route_question(question,router.memory)
            
            print("\n📊 Step 3: Final Results:")
            if not results:
                print("   ⚠️ No tools were triggered for this question.")
            else:
                print(json.dumps(results, indent=2))
                
        except Exception as e:
            print(f"\n❌ System Error: Something broke while running the tools!")
            print(f"   Details: {str(e)}")
            
        print("\n" + "="*50)

if __name__ == "__main__":
    run_interactive_test()
