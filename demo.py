#!/usr/bin/env python3
"""Streamlit demo for the Hiver support agent.

Run:
    .venv/bin/pip install streamlit
    .venv/bin/streamlit run demo.py
"""
import sys

import streamlit as st

sys.path.insert(0, '.')

from agent import Agent, sanitize_reply


@st.cache_resource(show_spinner=False)
def load_agent(engine):
    return Agent(engine=engine)


st.set_page_config(page_title='Hiver Support Agent', page_icon='💬', layout='wide')

st.title('Hiver Support Agent — @AmazonHelp')
st.caption(
    'Intent classification → history-grounded draft → auto-route or escalate to a human. '
    'No LLM required for inference; everything is retrieval + trained embeddings + a transparent rule.'
)

with st.sidebar:
    st.header('Agent config')
    engine = st.selectbox('Intent engine', ['tfidf', 'w2v'],
                          index=0, help='tfidf = char-level TF-IDF + LR (default, best); w2v = word2vec + LR')
    k = st.slider('Retrieval neighbours (k)', 1, 5, 3,
                  help='How many historical tweets to retrieve when grounding the draft.')
    author_id = st.text_input('Author id (optional)', value='',
                              help='Twitter author id of the customer, used for reply-history routing.')
    sample = st.selectbox('Sample tweet', [
        '',
        'my order never arrived',
        'still havnt got my package from last week',
        'thanks for the help!! appreciate it',
        'how do i contact amazon with my email',
        'amazon canceled my order can you refund me',
        '@amazonhelp order #112-4557890 delayed again',
    ])
    go = st.button('Run agent', type='primary', use_container_width=True)

st.header('Customer tweet')
if sample:
    text = st.text_area('Tweet', value=sample, height=90)
else:
    text = st.text_area('Tweet', value='my order never arrived', height=90)

if go:
    with st.spinner('Loading agent…'):
        agent = load_agent(engine)
    with st.spinner('Running agent…'):
        res = agent.respond(text, author_id=author_id or None, k=k)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric('Intent', res['pred_intent'], None)
        st.caption(f'confidence {res["pred_confidence"]:.4f}')
    with c2:
        route = res['route']
        st.metric('Route', f'{route.upper()}', None)
        st.caption(res['route_reason'])
    with c3:
        st.metric('Draft', 'send-ready' if (res['draft'] and sanitize_reply(res['draft']).strip()) else 'none',
                  None)

    st.divider()
    st.subheader(f'Draft reply ({res["route_reason"]})')
    if res['draft']:
        st.info(sanitize_reply(res['draft']))
    else:
        st.warning('No send-ready draft — escalate to a human agent.')

    if res['evidence']:
        st.subheader('Retrieval evidence')
        for e in res['evidence']:
            with st.expander(
                    f'sim {e["sim"]:.3f} · intent={e["intent"]} · '
                    f'customer tweet {e["customer_tweet_id"]}'):
                st.markdown(f'**Customer:** {e["customer_text"]}')
                st.markdown(f'**AmazonHelp replied:** {e["brand_reply_sanitized"]}')
    else:
        st.caption('No grounded retrieval found for this input.')

    st.divider()
    st.subheader('Routing decision')
    st.code(res['route_reason'])

    with st.expander('Show raw JSON'):
        import json as _json
        st.json({'intent': res['pred_intent'],
                 'confidence': res['pred_confidence'],
                 'route': res['route'],
                 'route_reason': res['route_reason'],
                 'draft': sanitize_reply(res['draft']),
                 'evidence': res['evidence']})