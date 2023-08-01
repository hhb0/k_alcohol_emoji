from typing import List, Generator
from openai.openai_object import OpenAIObject

import streamlit as st
st.set_page_config(
    page_title="k_tranditional_drink",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import openai
import time
import os
openai.api_key = st.secrets.OPENAI_TOKEN
from supabase import create_client
import pickle
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_c는 유능한 홍보 전문가입니다."},
        {"role": "user", "content": prompt}
    ],
    stream=True
)
    return response

def process_generated_text(streaming_resp: Generator[OpenAIObject, None, None]) -> str:
    report = []
    res_box = st.empty()
    for resp in streaming_resp:
        delta = resp.choices[0]["delta"]
        if "content" in delta:
            report.append(delta["content"])
            res_box.markdown("".join(report).strip())
        else:
            break
    result = "".join(report).strip()
    return result

@st.cache_resource(show_spinner=None, experimental_allow_widgets=True)
def get_idx_emoji(input_query, alcohol_min, alcohol_max):
    # 입력받은 쿼리 임베딩
    input_query_embedding = embedding_from_string(input_query, model=EMBEDDING_MODEL)

    # 임베딩 벡터간 거리 계산 (open ai 라이브러리 사용 - embeddings_utils.py)
    ## 도수 제한
    alcohol_limited_list = main_df.loc[
        (main_df["alcohol"] >= alcohol_min) & (main_df["alcohol"] <= alcohol_max)].index.tolist()
    source_embeddings = stacked_embeddings[alcohol_limited_list]

    distances = distances_from_embeddings(input_query_embedding, source_embeddings, distance_metric="cosine")

    # 가까운 벡터 인덱스 구하기 (open ai 라이브러리 사용 - embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)

    # 입력 받은 쿼리
    print(f"Query string: {input_query}")

    # k개의 가까운 벡터 인덱스 출력
    k_nearest_neighbors = 1
    k_counter = 0

    idx_list = []
    for i in indices_of_nearest_neighbors:
        # stop after printing out k articles
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1

        idx_list.append(i)

    return idx_list, alcohol_limited_list

def get_result(
        emotion: str,
        situation: str,
        ingredient: str,
        food: str,
        alcohol: str,
):

    if "\U0001F336" in ingredient or "\U0001F336" in food:
        ingredient = "\U0001F336"
        food = "\U0001F336"
    # query 수정
    situation_keyword = emoji_df[emoji_df["sample"] == situation]["k_keywords"].values[0]
    emotion_keyword = emoji_df[emoji_df["sample"] == emotion]["k_keywords"].values[0]
    ingredient_keyword = emoji_df[emoji_df["sample"] == ingredient]["k_keywords"].values[0]
    food_keyword = emoji_df[emoji_df["sample"] == food]["k_keywords"].values[0]

    input_query = f"재료는 {ingredient_keyword}다. 어울리는 음식으로는 {food_keyword}가 있다. {situation_keyword}다. {emotion_keyword} 감정을 언급할 수 있다."  # 벡터 임베딩용 쿼리
    result_query = f"{emotion} {situation} {ingredient} {food}"  # 출력용 쿼리

    # 알콜 이모지 도수로 변환
    if alcohol == "⬆️":
        alcohol_min = 18
        alcohol_max = 61

    else:
        alcohol_min = 0
        alcohol_max = 20

    idx_list, alcohol_limited_list = get_idx_emoji(input_query, alcohol_min, alcohol_max)

    name_id_list = []
    for i in idx_list:
        name_id_list.append(main_df.loc[alcohol_limited_list].iloc[i]["name_id"])

    # 결과 확인용
    print(f"{emotion}{situation}{food}로는 이게 딱!")

    for name_id in name_id_list:
        print(main_df[main_df["name_id"] == name_id]["name"].to_string(index=False))
        print(main_df[main_df["name_id"] == name_id]["alcohol"].to_string(index=False))
        print(feature_df[feature_df["name_id"] == name_id]["features"].to_string(index=False))
        print("---")

    return situation_keyword.split(",")[0], emotion_keyword.split(",")[0], ingredient_keyword.split(",")[0], result_query, name_id

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


def image_name(name_id):
    directory = "./f_image/"
    matching_files = [file for file in os.listdir(directory) if name_id in file]
    if len(matching_files) > 0:
        filename = os.path.join(directory, matching_files[0])
        return filename  # 변수 filename을 반환합니다.
    else:
        return None


input_container = None

@st.cache_resource(show_spinner=None, experimental_allow_widgets=True)
def write_propmt_result(emotion, situation, ingredient, food, name_id):
    supabase_client.table("result").insert(
        {
            "emotion": emotion,
            "situation": situation,
            "ingredient": ingredient,
            "food": food,
            "name_id": name_id,
        }
    ).execute()


with con2:
    container = st.empty()
    form = container.form("my_form", clear_on_submit=True)  # 내부 컨테이너의 폼 생성

    with form:
        empty7, col_s, empty9, col_e, empty8 = st.columns([0.05, 0.5, 0.2, 0.5, 0.05])
        with empty7:
            st.empty()

        with col_s:
            emotion = st.selectbox('감성', ('😁', '😭', '🥰', '😡', '😴', '🤢', '😱', '😎', '😂', '🥳'))

        with col_e:
            situation = st.selectbox("상황", ('☀️','☁️','❄️','🔥','☂️','💔','🎉','🎁','✈️','💍','💼','🚬','📝','💸','🌊','🌳','🍂','🌸','💪','👏','✌️','🙌','👍','👎'))

        with empty7:
            st.empty()

        empty10, col_i, empty15, col_f, empty11= st.columns([0.05, 0.5, 0.2, 0.5, 0.05])
        with empty10:
            st.empty()

        with col_i:
            ingredient = st.selectbox('재료', ('🍇','🍉','🍊','🍋','🍌','🍍','🍎','🍐','🍑','🍒','🍓','🍅','🌽','🌰','🥜',
                                             '🥔','🥕','🌶️','🍄','🌼','🎍','🌿','🍯','🥝','🥥','🌾','☕','🍵', '🍫','🍠','🧊','🥛'))

        with col_f:
            food = st.selectbox('어울리는 음식', ('🍕','🍔','🍟','🌭','🍿','🥞','🧈','🥐','🧀','🥗',
                                '🥩','🥟','🍤','🍱','🍚','🍜','🦪','🍣','🥘','🍝','🍦','🍩','🍪','🍰',
                                '🍫','🍬','🥛','🧃','🧊','🍯','🌶️','☕'))

        with empty11:
            st.empty()

        empty13, col_a, empty16, col_n, empty14= st.columns([0.05, 0.5, 0.2, 0.5, 0.05])
        with empty13:
            st.empty()
        with col_a:
            alcohol = st.selectbox('도수', ('⬇️','⬆️'))
        with empty16:
            st.empty()
        with col_n:
            real_name = st.text_input('이름 (선택)', placeholder="이름 또는 닉네임을 입력해주세요.")
        with empty14:
            st.empty()
        empty20,empty21,empty22,empty23,empty24,empty25 = st.columns(6)
        with empty25:
            submitted = st.form_submit_button("제출하기")

with st.container():  # 외부 컨테이너
    empty1, image_c, text_c, empty2 = st.columns([0.3, 0.3, 0.5, 0.3])
    name_id_list = []  # name_id_list 변수를 초기화합니다.
    if submitted:
        if not situation:
            st.error("어떤 상황에서 술을 마시고 싶은지 입력해주세요")
        elif not emotion:
            st.error("어떤 기분일 때 마시고 싶은지 입력해주세요")
        else:
            empty7, pro, empty9 = st.columns([0.3, 1.0, 0.3])
            with pro:
                with st.spinner('당신을 위한 전통주를 찾고 있습니다...🔍'):
                    situation_keyword, emotion_keyword, ingredient_keyword, result_query, name_id = get_result(situation=situation, emotion=emotion, food=food,
                                                                    ingredient=ingredient, alcohol=alcohol)
                    time.sleep(5)
                    if not name_id:
                        st.warning("검색 결과가 없습니다.")
                    else:
                        container.empty()
                        with image_c:
                            if name_id:
                                filtered_df = main_df[main_df["name_id"].str.contains(name_id)]
                                if not filtered_df.empty:
                                    loaded_image = image_name(name_id)
                                    st.image(loaded_image, use_column_width='auto')
                                else:
                                    st.write("해당하는 이미지가 없습니다.")

                        with text_c:
                            st.header(f"{emotion} {situation} {ingredient} {food}", anchor=False)
                            if real_name:
                                st.text(f"{real_name}님의 전통주 이모지 조합")
                            if name_id:
                                alcohol_name = main_df[main_df["name_id"]==name_id]["name"].to_string(index=False)
                                st.write(f"🔸 전통주 이름 : {alcohol_name}")
                                alcohol = main_df[main_df["name_id"] == name_id]["alcohol"].to_string(index=False)
                                st.write(f"🔸 도수 : {alcohol}")
                                st.write("🔸 특징 :")
                                features = feature_df[feature_df["name_id"] == name_id]["features"].to_string(index=False)
                                prompt = generate_prompt(name=alcohol_name, feature=features, situation_keyword=situation_keyword, emotion_keyword=emotion_keyword)
                                streaming_resp = request_chat_completion(prompt)
                                generated_text = process_generated_text(streaming_resp)
                                with_food = food_df[food_df["name_id"] == name_id]["food"].values[0]
                                st.write(f"🔸 어울리는 음식 : {with_food}")
                                if st.button('다시하기'):
                                    st.experimental_rerun()



                            else:
                                st.warning(f"전통주 이름: {name_id} 에 해당하는 정보가 없습니다.")





