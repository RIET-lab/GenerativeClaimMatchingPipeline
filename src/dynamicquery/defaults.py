# paths for clef checkthat task 2a english
TASK_2A_EN_PATH = "clef2022-checkthat-lab/task2/data/subtask-2a--english"

TASK_2A_EN_TRAIN_QUERY_PATH = TASK_2A_EN_PATH + "/CT2022-Task2A-EN-Train-Dev_Queries.tsv"
TASK_2A_EN_DEV_QUERY_PATH = TASK_2A_EN_PATH + "/CT2022-Task2A-EN-Train-Dev_Queries.tsv"
TASK_2A_EN_TEST21_QUERY_PATH = TASK_2A_EN_PATH + "/CT2022-Task2A-EN-Dev-Test_Queries.tsv"
TASK_2A_EN_TEST22_QUERY_PATH = TASK_2A_EN_PATH + "/test/CT2022-Task2A-EN-Test_Queries.tsv"

TASK_2A_EN_TARGETS_PATH = TASK_2A_EN_PATH + "/vclaims"
TASK_2A_EN_TARGETS_KEY_NAMES = ["title", "subtitle", "author", "date","target_id", "target", "page_url"]

TASK_2A_EN_TRAIN_QREL_PATH = TASK_2A_EN_PATH + "/CT2022-Task2A-EN-Train_QRELs.tsv"
TASK_2A_EN_DEV_QREL_PATH = TASK_2A_EN_PATH + "/CT2022-Task2A-EN-Dev_QRELs.tsv"
TASK_2A_EN_TEST21_QREL_PATH = TASK_2A_EN_PATH + "/CT2022-Task2A-EN-Dev-Test_QRELs.tsv"

TASK_2A_EN_QUERY_PREFIX = "tweet: "
TASK_2A_EN_TARGET_PREFIX = "claim: "

# paths for clef checkthat task 2b english
TASK_2B_EN_PATH = "clef2022-checkthat-lab/task2/data/subtask-2b--english"

TASK_2B_EN_TRAIN_QUERY_PATH = TASK_2B_EN_PATH + "/_CT2022-Task2B-EN-Train-Dev_Queries.tsv"
TASK_2B_EN_DEV_QUERY_PATH = TASK_2B_EN_PATH + "/_CT2022-Task2B-EN-Train-Dev_Queries.tsv"
TASK_2B_EN_TEST21_QUERY_PATH = TASK_2B_EN_PATH + "/_CT2022-Task2B-EN-Dev-Test_Queries.tsv"
TASK_2B_EN_TEST22_QUERY_PATH = TASK_2B_EN_PATH + "/test/_CT2022-Task2B-EN-Test_Queries.tsv"

TASK_2B_EN_TARGETS_PATH = TASK_2B_EN_PATH + "/politifact-vclaims"
TASK_2B_EN_TARGETS_KEY_NAMES = ["url", "speaker", "target", "truth_label", "date", "title", "text", "target_id"]

TASK_2B_EN_TRAIN_QREL_PATH = TASK_2B_EN_PATH + "/CT2022-Task2B-EN-Train_QRELs.tsv"
TASK_2B_EN_DEV_QREL_PATH = TASK_2B_EN_PATH + "/CT2022-Task2B-EN-Dev_QRELs.tsv"
TASK_2B_EN_TEST21_QREL_PATH = TASK_2B_EN_PATH + "/CT2022-Task2B-EN-Dev-Test_QRELs.tsv"

TASK_2B_EN_QUERY_PREFIX = "statement: "
TASK_2B_EN_TARGET_PREFIX = "claim: "

# paths for clef checkthat task 2a arabic
TASK_2A_AR_PATH = "clef2022-checkthat-lab/task2/data/subtask-2a--arabic"
TASK_2A_AR_TRANSLATED_PATH = "CheckThat-Arabic"


TASK_2A_AR_TRAIN_QUERY_PATH = TASK_2A_AR_TRANSLATED_PATH + "/english_CT2022-Task2A-AR-Train_Queries.txt"
TASK_2A_AR_DEV_QUERY_PATH = TASK_2A_AR_TRANSLATED_PATH + "/english_CT2022-Task2A-AR-Dev_Queries.txt"
TASK_2A_AR_TEST21_QUERY_PATH = TASK_2A_AR_TRANSLATED_PATH + "/english_CT2022-Task2A-AR-Dev-Test_Queries.tsv"
TASK_2A_AR_TEST22_QUERY_PATH = TASK_2A_AR_TRANSLATED_PATH + "/test/english_CT2022-Task2A-AR-Test_Queries.tsv"

TASK_2A_AR_TARGETS_PATH = TASK_2A_AR_TRANSLATED_PATH + "/en_vclaims_formatted"
TASK_2A_AR_TARGETS_KEY_NAMES = ["target_id", "target", "title"]

TASK_2A_AR_TRAIN_QREL_PATH = TASK_2A_AR_PATH + "/CT2022-Task2A-AR-Train_QRELs.txt"
TASK_2A_AR_DEV_QREL_PATH = TASK_2A_AR_PATH + "/CT2022-Task2A-AR-Dev_QRELs.txt"
TASK_2A_AR_TEST21_QREL_PATH = TASK_2A_AR_PATH + "/CT2022-Task2A-AR-Dev-Test_QRELs.tsv"

TASK_2A_AR_QUERY_PREFIX = "tweet: "
TASK_2A_AR_TARGET_PREFIX = "claim: "