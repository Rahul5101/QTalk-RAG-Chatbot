from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
import re
import os
from dotenv import load_dotenv

load_dotenv()

SERVICE_ACCOUNT_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service-account.json")

translate_client = None
if os.path.isfile(SERVICE_ACCOUNT_PATH):
    try:
        creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH)
        translate_client = translate.Client(credentials=creds)
    except Exception as exc:
        print(f"⚠️ Google Translate client init warning: {exc}")


def translation(detected_lang, user_query):
    if not user_query:
        return ""
    # Translate non-English query to English if client is available
    if detected_lang != "en" and translate_client is not None:
        try:
            result = translate_client.translate(user_query, target_language="en")
            return result.get("translatedText", user_query)
        except Exception as e:
            print(f"[Translation Error] {e}. Falling back to raw user query.")
            return user_query
    return user_query


def output_converison(text, targeted_language):
    if not text:
        return ""
    if targeted_language and targeted_language.strip() and translate_client is not None:
        try:
            placeholder = "__[[[LINE_BREAK]]]__"
            text_with_placeholders = text.replace("\n", placeholder)

            result = translate_client.translate(
                text_with_placeholders,
                target_language=targeted_language
            )

            translated_text = result['translatedText']

            translated_text = re.sub(
                r'[_\s\[\]]*LINE[\s_]*BREAK[_\s\[\]]*',
                placeholder,
                translated_text,
                flags=re.IGNORECASE
            )

            text = translated_text.replace(placeholder, "\n")

        except Exception as e:
            print(f"Translation output failed: {e}")
    return text
