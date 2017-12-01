# R Library Analysis

Shinyapp:     <a href=https://mengyingliu.shinyapps.io/shinyrlib/>https://mengyingliu.shinyapps.io/shinyrlib/</a>

GitHub Analysis data source: <a href=https://api.github.com/>https://api.github.com/</a>

<h1>Group Member</h1>
<p>Rong Wang</p>
<p>Mengying Liu</p>
<p>Yi Liu</p>
<p>Zhibo Wan</p>
<p>Weihan Li</p>

<h1>Please run this code on your laptop so you could use the API of github for analysis</h1>
<p>oauth_endpoints("github")</p>
<p>myapp <- oauth_app("github",
                   key = "19c8034e916ab8dd7f3c",
                   secret = "a54ce3798c0ed7c188d87ddb8a2b61d1951b7935")</p>
<p>github_token <- oauth2.0_token(oauth_endpoints("github"), myapp)</p>
<p>#change to reactive</p>
<p>gtoken <- config(token = github_token)</p>
