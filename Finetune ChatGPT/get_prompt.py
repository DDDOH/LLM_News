SYSTEM = "You are an editor tasked with choosing the catchier one from several drafted headlines for the same article. Catchier means the one that is likely to generate more clicks."
USER = "You are presented with several headlines. Which one is catchier? **Return only the number before the headline. **No explanation is needed. No need to return the headline, only the number.****\n"


def prompt_with_label(headlines, best_id):
    user_message = USER
    for i in range(len(headlines)):
        user_message += f"{i+1}. {headlines[i]}\n"
    
    recording = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": str(best_id)}
        ]
    }
    return recording

def prompt_without_label(headlines):
    user_message = USER
    for i in range(len(headlines)):
        user_message += f"{i+1}. {headlines[i]}\n"
    
    recording = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message},
        ]
    return recording