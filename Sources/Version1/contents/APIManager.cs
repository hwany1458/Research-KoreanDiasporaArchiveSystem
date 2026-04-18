/*
 * APIManager.cs
 * 
 * REST API 통신 관리자
 * Python FastAPI 서버와의 HTTP 통신 처리
 */

using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;
using Diaspora.Common;

namespace Diaspora.API
{
    /// <summary>
    /// API 통신 관리자
    /// </summary>
    public class APIManager : Singleton<APIManager>
    {
        [Header("Settings")]
        [SerializeField] private string baseUrl = APIConstants.BASE_URL;
        [SerializeField] private int requestTimeout = APIConstants.REQUEST_TIMEOUT;
        [SerializeField] private bool logRequests = true;

        // 이벤트
        public event Action<string> OnRequestStarted;
        public event Action<string, bool> OnRequestCompleted;
        public event Action<string, string> OnRequestError;

        // 캐시
        private Dictionary<string, string> responseCache = new Dictionary<string, string>();
        private Dictionary<string, float> cacheTimestamps = new Dictionary<string, float>();

        protected override void OnSingletonAwake()
        {
            // 개발 환경에서 URL 설정
#if UNITY_EDITOR
            baseUrl = "http://localhost:8000";
#endif
        }

        #region GET 요청

        /// <summary>
        /// GET 요청 (코루틴)
        /// </summary>
        public IEnumerator GetRequest<T>(string endpoint, Action<T> onSuccess, Action<string> onError, bool useCache = true) where T : class
        {
            string url = baseUrl + endpoint;
            
            // 캐시 확인
            if (useCache && TryGetCache(url, out string cachedResponse))
            {
                T cachedData = JsonUtility.FromJson<T>(cachedResponse);
                onSuccess?.Invoke(cachedData);
                yield break;
            }

            OnRequestStarted?.Invoke(url);
            if (logRequests) Debug.Log($"[API] GET: {url}");

            using (UnityWebRequest request = UnityWebRequest.Get(url))
            {
                request.timeout = requestTimeout;
                request.SetRequestHeader("Content-Type", "application/json");

                yield return request.SendWebRequest();

                if (request.result == UnityWebRequest.Result.Success)
                {
                    string response = request.downloadHandler.text;
                    
                    // 캐시 저장
                    if (useCache) SetCache(url, response);

                    try
                    {
                        T data = JsonUtility.FromJson<T>(response);
                        onSuccess?.Invoke(data);
                        OnRequestCompleted?.Invoke(url, true);
                    }
                    catch (Exception e)
                    {
                        string error = $"JSON 파싱 오류: {e.Message}";
                        onError?.Invoke(error);
                        OnRequestError?.Invoke(url, error);
                    }
                }
                else
                {
                    string error = $"요청 실패: {request.error}";
                    onError?.Invoke(error);
                    OnRequestError?.Invoke(url, error);
                    OnRequestCompleted?.Invoke(url, false);
                }
            }
        }

        /// <summary>
        /// GET 요청 (async/await)
        /// </summary>
        public async Task<T> GetAsync<T>(string endpoint, bool useCache = true) where T : class
        {
            string url = baseUrl + endpoint;

            // 캐시 확인
            if (useCache && TryGetCache(url, out string cachedResponse))
            {
                return JsonUtility.FromJson<T>(cachedResponse);
            }

            OnRequestStarted?.Invoke(url);
            if (logRequests) Debug.Log($"[API] GET: {url}");

            using (UnityWebRequest request = UnityWebRequest.Get(url))
            {
                request.timeout = requestTimeout;
                request.SetRequestHeader("Content-Type", "application/json");

                var operation = request.SendWebRequest();
                while (!operation.isDone)
                    await Task.Yield();

                if (request.result == UnityWebRequest.Result.Success)
                {
                    string response = request.downloadHandler.text;
                    if (useCache) SetCache(url, response);
                    OnRequestCompleted?.Invoke(url, true);
                    return JsonUtility.FromJson<T>(response);
                }
                else
                {
                    string error = $"요청 실패: {request.error}";
                    OnRequestError?.Invoke(url, error);
                    OnRequestCompleted?.Invoke(url, false);
                    throw new Exception(error);
                }
            }
        }

        #endregion

        #region POST 요청

        /// <summary>
        /// POST 요청 (코루틴)
        /// </summary>
        public IEnumerator PostRequest<TRequest, TResponse>(string endpoint, TRequest data, Action<TResponse> onSuccess, Action<string> onError) 
            where TRequest : class 
            where TResponse : class
        {
            string url = baseUrl + endpoint;
            string jsonData = JsonUtility.ToJson(data);

            OnRequestStarted?.Invoke(url);
            if (logRequests) Debug.Log($"[API] POST: {url}\nBody: {jsonData}");

            using (UnityWebRequest request = new UnityWebRequest(url, "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);
                request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.timeout = requestTimeout;
                request.SetRequestHeader("Content-Type", "application/json");

                yield return request.SendWebRequest();

                if (request.result == UnityWebRequest.Result.Success)
                {
                    string response = request.downloadHandler.text;
                    try
                    {
                        TResponse responseData = JsonUtility.FromJson<TResponse>(response);
                        onSuccess?.Invoke(responseData);
                        OnRequestCompleted?.Invoke(url, true);
                    }
                    catch (Exception e)
                    {
                        string error = $"JSON 파싱 오류: {e.Message}";
                        onError?.Invoke(error);
                        OnRequestError?.Invoke(url, error);
                    }
                }
                else
                {
                    string error = $"요청 실패: {request.error}";
                    onError?.Invoke(error);
                    OnRequestError?.Invoke(url, error);
                    OnRequestCompleted?.Invoke(url, false);
                }
            }
        }

        /// <summary>
        /// POST 요청 (async/await)
        /// </summary>
        public async Task<TResponse> PostAsync<TRequest, TResponse>(string endpoint, TRequest data) 
            where TRequest : class 
            where TResponse : class
        {
            string url = baseUrl + endpoint;
            string jsonData = JsonUtility.ToJson(data);

            OnRequestStarted?.Invoke(url);
            if (logRequests) Debug.Log($"[API] POST: {url}");

            using (UnityWebRequest request = new UnityWebRequest(url, "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);
                request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.timeout = requestTimeout;
                request.SetRequestHeader("Content-Type", "application/json");

                var operation = request.SendWebRequest();
                while (!operation.isDone)
                    await Task.Yield();

                if (request.result == UnityWebRequest.Result.Success)
                {
                    string response = request.downloadHandler.text;
                    OnRequestCompleted?.Invoke(url, true);
                    return JsonUtility.FromJson<TResponse>(response);
                }
                else
                {
                    string error = $"요청 실패: {request.error}";
                    OnRequestError?.Invoke(url, error);
                    OnRequestCompleted?.Invoke(url, false);
                    throw new Exception(error);
                }
            }
        }

        #endregion

        #region 파일 다운로드

        /// <summary>
        /// 이미지 다운로드
        /// </summary>
        public IEnumerator DownloadImage(string imageUrl, Action<Texture2D> onSuccess, Action<string> onError)
        {
            string url = imageUrl.StartsWith("http") ? imageUrl : baseUrl + imageUrl;

            if (logRequests) Debug.Log($"[API] Download Image: {url}");

            using (UnityWebRequest request = UnityWebRequestTexture.GetTexture(url))
            {
                request.timeout = APIConstants.DOWNLOAD_TIMEOUT;

                yield return request.SendWebRequest();

                if (request.result == UnityWebRequest.Result.Success)
                {
                    Texture2D texture = DownloadHandlerTexture.GetContent(request);
                    onSuccess?.Invoke(texture);
                }
                else
                {
                    onError?.Invoke($"이미지 다운로드 실패: {request.error}");
                }
            }
        }

        /// <summary>
        /// 오디오 다운로드
        /// </summary>
        public IEnumerator DownloadAudio(string audioUrl, AudioType audioType, Action<AudioClip> onSuccess, Action<string> onError)
        {
            string url = audioUrl.StartsWith("http") ? audioUrl : baseUrl + audioUrl;

            if (logRequests) Debug.Log($"[API] Download Audio: {url}");

            using (UnityWebRequest request = UnityWebRequestMultimedia.GetAudioClip(url, audioType))
            {
                request.timeout = APIConstants.DOWNLOAD_TIMEOUT;

                yield return request.SendWebRequest();

                if (request.result == UnityWebRequest.Result.Success)
                {
                    AudioClip clip = DownloadHandlerAudioClip.GetContent(request);
                    onSuccess?.Invoke(clip);
                }
                else
                {
                    onError?.Invoke($"오디오 다운로드 실패: {request.error}");
                }
            }
        }

        #endregion

        #region API 엔드포인트 메서드

        /// <summary>
        /// 자료 목록 조회
        /// </summary>
        public IEnumerator GetItems(int page, int pageSize, Action<ItemListResponse> onSuccess, Action<string> onError)
        {
            string endpoint = $"{APIConstants.ITEMS_ENDPOINT}?page={page}&page_size={pageSize}";
            yield return GetRequest(endpoint, onSuccess, onError);
        }

        /// <summary>
        /// 자료 상세 조회
        /// </summary>
        public IEnumerator GetItem(string itemId, Action<ItemData> onSuccess, Action<string> onError)
        {
            string endpoint = $"{APIConstants.ITEMS_ENDPOINT}/{itemId}";
            yield return GetRequest(endpoint, onSuccess, onError);
        }

        /// <summary>
        /// 인물 목록 조회
        /// </summary>
        public IEnumerator GetPersons(int page, int pageSize, Action<PersonListResponse> onSuccess, Action<string> onError)
        {
            string endpoint = $"{APIConstants.PERSONS_ENDPOINT}?page={page}&page_size={pageSize}";
            yield return GetRequest(endpoint, onSuccess, onError);
        }

        /// <summary>
        /// 인물 상세 조회
        /// </summary>
        public IEnumerator GetPerson(string personId, Action<PersonData> onSuccess, Action<string> onError)
        {
            string endpoint = $"{APIConstants.PERSONS_ENDPOINT}/{personId}";
            yield return GetRequest(endpoint, onSuccess, onError);
        }

        /// <summary>
        /// 검색
        /// </summary>
        public IEnumerator Search(SearchRequest searchRequest, Action<SearchResponse> onSuccess, Action<string> onError)
        {
            yield return PostRequest<SearchRequest, SearchResponse>(APIConstants.SEARCH_ENDPOINT, searchRequest, onSuccess, onError);
        }

        /// <summary>
        /// 타임라인 조회
        /// </summary>
        public IEnumerator GetTimeline(int? yearFrom, int? yearTo, Action<TimelineResponse> onSuccess, Action<string> onError)
        {
            string endpoint = APIConstants.TIMELINE_ENDPOINT;
            List<string> queryParams = new List<string>();
            
            if (yearFrom.HasValue) queryParams.Add($"year_from={yearFrom.Value}");
            if (yearTo.HasValue) queryParams.Add($"year_to={yearTo.Value}");
            
            if (queryParams.Count > 0)
                endpoint += "?" + string.Join("&", queryParams);
            
            yield return GetRequest(endpoint, onSuccess, onError);
        }

        /// <summary>
        /// 관계 그래프 조회
        /// </summary>
        public IEnumerator GetGraph(string centerId, int depth, Action<GraphResponse> onSuccess, Action<string> onError)
        {
            string endpoint = $"{APIConstants.GRAPH_ENDPOINT}?center_id={centerId}&depth={depth}";
            yield return GetRequest(endpoint, onSuccess, onError);
        }

        /// <summary>
        /// 관련 자료 조회
        /// </summary>
        public IEnumerator GetRelatedItems(string itemId, int limit, Action<RelatedItemsResponse> onSuccess, Action<string> onError)
        {
            string endpoint = $"{APIConstants.GRAPH_ENDPOINT}/related/{itemId}?limit={limit}";
            yield return GetRequest(endpoint, onSuccess, onError);
        }

        /// <summary>
        /// 스토리 조회
        /// </summary>
        public IEnumerator GetStory(string storyId, Action<StoryData> onSuccess, Action<string> onError)
        {
            string endpoint = $"{APIConstants.STORIES_ENDPOINT}/{storyId}";
            yield return GetRequest(endpoint, onSuccess, onError);
        }

        /// <summary>
        /// 통계 조회
        /// </summary>
        public IEnumerator GetStats(Action<StatsResponse> onSuccess, Action<string> onError)
        {
            yield return GetRequest("/stats", onSuccess, onError);
        }

        #endregion

        #region 캐시 관리

        private bool TryGetCache(string key, out string value)
        {
            if (responseCache.TryGetValue(key, out value))
            {
                if (cacheTimestamps.TryGetValue(key, out float timestamp))
                {
                    if (Time.time - timestamp < CacheConstants.METADATA_CACHE_DURATION)
                    {
                        return true;
                    }
                }
                // 만료된 캐시 제거
                responseCache.Remove(key);
                cacheTimestamps.Remove(key);
            }
            value = null;
            return false;
        }

        private void SetCache(string key, string value)
        {
            responseCache[key] = value;
            cacheTimestamps[key] = Time.time;
        }

        /// <summary>
        /// 캐시 클리어
        /// </summary>
        public void ClearCache()
        {
            responseCache.Clear();
            cacheTimestamps.Clear();
            Debug.Log("[API] 캐시가 클리어되었습니다.");
        }

        #endregion

        #region 유틸리티

        /// <summary>
        /// 서버 연결 확인
        /// </summary>
        public IEnumerator CheckConnection(Action<bool> callback)
        {
            string url = baseUrl + "/health";

            using (UnityWebRequest request = UnityWebRequest.Get(url))
            {
                request.timeout = 5;
                yield return request.SendWebRequest();
                callback?.Invoke(request.result == UnityWebRequest.Result.Success);
            }
        }

        /// <summary>
        /// Base URL 설정
        /// </summary>
        public void SetBaseUrl(string url)
        {
            baseUrl = url;
            ClearCache();
            Debug.Log($"[API] Base URL 변경: {url}");
        }

        #endregion
    }
}
