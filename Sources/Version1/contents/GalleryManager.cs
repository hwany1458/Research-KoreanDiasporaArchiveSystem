/*
 * GalleryManager.cs
 * 
 * VR 갤러리 관리자
 * 전시 공간 구성 및 자료 배치
 */

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Diaspora.Common;
using Diaspora.API;

namespace Diaspora.VR
{
    /// <summary>
    /// 갤러리 관리자
    /// </summary>
    public class GalleryManager : Singleton<GalleryManager>
    {
        [Header("Prefabs")]
        [SerializeField] private GameObject exhibitFramePrefab;
        [SerializeField] private GameObject infoStandPrefab;
        [SerializeField] private GameObject teleportPointPrefab;
        [SerializeField] private GameObject roomDividerPrefab;

        [Header("Gallery Settings")]
        [SerializeField] private float roomWidth = 15f;
        [SerializeField] private float roomHeight = 4f;
        [SerializeField] private float roomDepth = 15f;
        [SerializeField] private float exhibitSpacing = 2.5f;
        [SerializeField] private float exhibitHeight = 1.5f;
        [SerializeField] private float wallOffset = 0.1f;

        [Header("Layout")]
        [SerializeField] private LayoutMode layoutMode = LayoutMode.Timeline;
        [SerializeField] private int maxExhibitsPerWall = 5;

        [Header("References")]
        [SerializeField] private Transform galleryRoot;
        [SerializeField] private Transform exhibitsParent;

        // 생성된 오브젝트
        private List<ExhibitItem> exhibits = new List<ExhibitItem>();
        private List<GameObject> teleportPoints = new List<GameObject>();
        private Dictionary<string, ExhibitItem> exhibitLookup = new Dictionary<string, ExhibitItem>();

        // 데이터
        private List<ItemData> loadedItems = new List<ItemData>();
        private StoryData currentStory;

        // 이벤트
        public event Action OnGalleryLoaded;
        public event Action<ExhibitItem> OnExhibitSelected;
        public event Action<string> OnLoadError;

        // 프로퍼티
        public int ExhibitCount => exhibits.Count;
        public List<ExhibitItem> Exhibits => exhibits;

        /// <summary>
        /// 레이아웃 모드
        /// </summary>
        public enum LayoutMode
        {
            Timeline,       // 연대순 배치
            Grid,          // 격자 배치
            Circular,      // 원형 배치
            Custom         // 커스텀 배치
        }

        protected override void OnSingletonAwake()
        {
            if (galleryRoot == null)
                galleryRoot = transform;
            if (exhibitsParent == null)
                exhibitsParent = galleryRoot;
        }

        #region 갤러리 로드

        /// <summary>
        /// 자료 목록으로 갤러리 구성
        /// </summary>
        public void LoadGallery(List<ItemData> items)
        {
            loadedItems = items;
            StartCoroutine(LoadGalleryRoutine(items));
        }

        /// <summary>
        /// 스토리 기반 갤러리 구성
        /// </summary>
        public void LoadGalleryFromStory(StoryData story)
        {
            currentStory = story;
            // 스토리 세그먼트의 자료들을 로드
            List<string> itemIds = new List<string>();
            foreach (var segment in story.segments)
            {
                if (segment.item_ids != null)
                {
                    itemIds.AddRange(segment.item_ids);
                }
            }

            StartCoroutine(LoadGalleryFromItemIds(itemIds));
        }

        /// <summary>
        /// 서버에서 자료 로드
        /// </summary>
        public void LoadGalleryFromServer(int page = 1, int pageSize = 20)
        {
            StartCoroutine(LoadFromServerRoutine(page, pageSize));
        }

        private IEnumerator LoadFromServerRoutine(int page, int pageSize)
        {
            yield return APIManager.Instance.GetItems(page, pageSize,
                (ItemListResponse response) =>
                {
                    LoadGallery(response.items);
                },
                (string error) =>
                {
                    OnLoadError?.Invoke(error);
                    Debug.LogError($"[Gallery] 로드 실패: {error}");
                }
            );
        }

        private IEnumerator LoadGalleryFromItemIds(List<string> itemIds)
        {
            List<ItemData> items = new List<ItemData>();

            foreach (string itemId in itemIds)
            {
                yield return APIManager.Instance.GetItem(itemId,
                    (ItemData item) => items.Add(item),
                    (string error) => Debug.LogWarning($"[Gallery] 자료 로드 실패: {itemId}")
                );
            }

            LoadGallery(items);
        }

        private IEnumerator LoadGalleryRoutine(List<ItemData> items)
        {
            // 기존 전시물 제거
            ClearGallery();

            // 전시물 배치
            switch (layoutMode)
            {
                case LayoutMode.Timeline:
                    yield return ArrangeTimeline(items);
                    break;
                case LayoutMode.Grid:
                    yield return ArrangeGrid(items);
                    break;
                case LayoutMode.Circular:
                    yield return ArrangeCircular(items);
                    break;
                default:
                    yield return ArrangeTimeline(items);
                    break;
            }

            // 텔레포트 포인트 생성
            CreateTeleportPoints();

            // 이미지 로드
            yield return LoadExhibitImages();

            OnGalleryLoaded?.Invoke();
            Debug.Log($"[Gallery] 갤러리 로드 완료: {exhibits.Count}개 전시물");
        }

        #endregion

        #region 레이아웃 배치

        /// <summary>
        /// 타임라인 배치 (연대순)
        /// </summary>
        private IEnumerator ArrangeTimeline(List<ItemData> items)
        {
            // 날짜순 정렬
            items.Sort((a, b) => string.Compare(a.date ?? "", b.date ?? ""));

            // 연도별 그룹화
            Dictionary<int, List<ItemData>> yearGroups = new Dictionary<int, List<ItemData>>();
            foreach (var item in items)
            {
                int year = ParseYear(item.date);
                int decade = (year / 10) * 10;
                if (!yearGroups.ContainsKey(decade))
                {
                    yearGroups[decade] = new List<ItemData>();
                }
                yearGroups[decade].Add(item);
            }

            // 연대별 벽면 배치
            int wallIndex = 0;
            foreach (var kvp in yearGroups)
            {
                int decade = kvp.Key;
                List<ItemData> decadeItems = kvp.Value;

                // 벽면 위치 계산
                Vector3 wallPosition = GetWallPosition(wallIndex);
                Quaternion wallRotation = GetWallRotation(wallIndex);

                // 연대 표지판 생성
                CreateDecadeSign(decade, wallPosition, wallRotation);

                // 전시물 배치
                for (int i = 0; i < decadeItems.Count && i < maxExhibitsPerWall; i++)
                {
                    Vector3 exhibitPos = CalculateExhibitPosition(wallPosition, wallRotation, i, decadeItems.Count);
                    CreateExhibit(decadeItems[i], exhibitPos, wallRotation);
                    yield return null; // 프레임 분산
                }

                wallIndex++;
            }
        }

        /// <summary>
        /// 격자 배치
        /// </summary>
        private IEnumerator ArrangeGrid(List<ItemData> items)
        {
            int columns = Mathf.CeilToInt(Mathf.Sqrt(items.Count));
            int rows = Mathf.CeilToInt((float)items.Count / columns);

            for (int i = 0; i < items.Count; i++)
            {
                int col = i % columns;
                int row = i / columns;

                float x = (col - columns / 2f) * exhibitSpacing;
                float z = (row - rows / 2f) * exhibitSpacing;
                Vector3 position = new Vector3(x, exhibitHeight, z);

                CreateExhibit(items[i], position, Quaternion.identity);
                yield return null;
            }
        }

        /// <summary>
        /// 원형 배치
        /// </summary>
        private IEnumerator ArrangeCircular(List<ItemData> items)
        {
            float radius = Mathf.Max(roomWidth, roomDepth) / 2f - 1f;
            float angleStep = 360f / items.Count;

            for (int i = 0; i < items.Count; i++)
            {
                float angle = i * angleStep * Mathf.Deg2Rad;
                float x = Mathf.Sin(angle) * radius;
                float z = Mathf.Cos(angle) * radius;
                Vector3 position = new Vector3(x, exhibitHeight, z);

                // 중앙을 향하도록 회전
                Quaternion rotation = Quaternion.LookRotation(-position.normalized);

                CreateExhibit(items[i], position, rotation);
                yield return null;
            }
        }

        /// <summary>
        /// 벽면 위치 계산
        /// </summary>
        private Vector3 GetWallPosition(int wallIndex)
        {
            // 4면 벽 (북, 동, 남, 서)
            switch (wallIndex % 4)
            {
                case 0: return new Vector3(0, exhibitHeight, roomDepth / 2 - wallOffset);
                case 1: return new Vector3(roomWidth / 2 - wallOffset, exhibitHeight, 0);
                case 2: return new Vector3(0, exhibitHeight, -roomDepth / 2 + wallOffset);
                case 3: return new Vector3(-roomWidth / 2 + wallOffset, exhibitHeight, 0);
                default: return Vector3.zero;
            }
        }

        /// <summary>
        /// 벽면 회전 계산
        /// </summary>
        private Quaternion GetWallRotation(int wallIndex)
        {
            switch (wallIndex % 4)
            {
                case 0: return Quaternion.Euler(0, 180, 0);
                case 1: return Quaternion.Euler(0, -90, 0);
                case 2: return Quaternion.Euler(0, 0, 0);
                case 3: return Quaternion.Euler(0, 90, 0);
                default: return Quaternion.identity;
            }
        }

        /// <summary>
        /// 전시물 위치 계산
        /// </summary>
        private Vector3 CalculateExhibitPosition(Vector3 wallCenter, Quaternion wallRotation, int index, int total)
        {
            float offset = (index - (total - 1) / 2f) * exhibitSpacing;
            Vector3 localOffset = new Vector3(offset, 0, 0);
            return wallCenter + wallRotation * localOffset;
        }

        #endregion

        #region 전시물 생성

        /// <summary>
        /// 전시물 생성
        /// </summary>
        private ExhibitItem CreateExhibit(ItemData item, Vector3 position, Quaternion rotation)
        {
            if (exhibitFramePrefab == null)
            {
                Debug.LogError("[Gallery] exhibitFramePrefab이 설정되지 않았습니다.");
                return null;
            }

            GameObject exhibitObj = Instantiate(exhibitFramePrefab, exhibitsParent);
            exhibitObj.transform.position = position;
            exhibitObj.transform.rotation = rotation;
            exhibitObj.name = $"Exhibit_{item.item_id}";

            ExhibitItem exhibit = exhibitObj.GetComponent<ExhibitItem>();
            if (exhibit == null)
            {
                exhibit = exhibitObj.AddComponent<ExhibitItem>();
            }

            exhibit.Initialize(item);
            exhibit.OnSelected += HandleExhibitSelected;

            exhibits.Add(exhibit);
            exhibitLookup[item.item_id] = exhibit;

            return exhibit;
        }

        /// <summary>
        /// 연대 표지판 생성
        /// </summary>
        private void CreateDecadeSign(int decade, Vector3 position, Quaternion rotation)
        {
            // 벽 상단에 연대 표시
            Vector3 signPosition = position + Vector3.up * 1.5f;
            
            // 간단한 텍스트 메시 생성 (실제로는 프리팹 사용)
            GameObject sign = new GameObject($"DecadeSign_{decade}");
            sign.transform.SetParent(exhibitsParent);
            sign.transform.position = signPosition;
            sign.transform.rotation = rotation;

            var textMesh = sign.AddComponent<TextMesh>();
            textMesh.text = $"{decade}년대";
            textMesh.fontSize = 50;
            textMesh.alignment = TextAlignment.Center;
            textMesh.anchor = TextAnchor.MiddleCenter;
            textMesh.characterSize = 0.05f;
            textMesh.color = Color.white;
        }

        /// <summary>
        /// 정보 안내대 생성
        /// </summary>
        private void CreateInfoStand(ItemData item, Vector3 position)
        {
            if (infoStandPrefab == null) return;

            GameObject stand = Instantiate(infoStandPrefab, exhibitsParent);
            stand.transform.position = position + Vector3.forward * 0.5f + Vector3.down * 0.5f;
            stand.transform.LookAt(position);
        }

        #endregion

        #region 텔레포트 포인트

        /// <summary>
        /// 텔레포트 포인트 생성
        /// </summary>
        private void CreateTeleportPoints()
        {
            if (teleportPointPrefab == null) return;

            // 갤러리 중앙
            CreateTeleportPoint(Vector3.zero, "중앙");

            // 각 벽면 앞
            for (int i = 0; i < 4; i++)
            {
                Vector3 wallPos = GetWallPosition(i);
                Vector3 viewPos = wallPos - GetWallRotation(i) * Vector3.forward * 3f;
                viewPos.y = 0;
                CreateTeleportPoint(viewPos, $"구역 {i + 1}");
            }
        }

        private void CreateTeleportPoint(Vector3 position, string label)
        {
            GameObject point = Instantiate(teleportPointPrefab, galleryRoot);
            point.transform.position = position;
            point.name = $"TeleportPoint_{label}";

            // 라벨 설정 (프리팹에 컴포넌트가 있다면)
            var labelComponent = point.GetComponentInChildren<TextMesh>();
            if (labelComponent != null)
            {
                labelComponent.text = label;
            }

            teleportPoints.Add(point);
        }

        #endregion

        #region 이미지 로드

        /// <summary>
        /// 전시물 이미지 로드
        /// </summary>
        private IEnumerator LoadExhibitImages()
        {
            foreach (var exhibit in exhibits)
            {
                if (exhibit != null && !string.IsNullOrEmpty(exhibit.ItemData?.thumbnail_url))
                {
                    yield return APIManager.Instance.DownloadImage(
                        exhibit.ItemData.thumbnail_url,
                        (Texture2D texture) => exhibit.SetImage(texture),
                        (string error) => Debug.LogWarning($"[Gallery] 이미지 로드 실패: {error}")
                    );
                }
            }
        }

        #endregion

        #region 이벤트 핸들러

        private void HandleExhibitSelected(ExhibitItem exhibit)
        {
            OnExhibitSelected?.Invoke(exhibit);
        }

        #endregion

        #region 유틸리티

        /// <summary>
        /// 갤러리 초기화
        /// </summary>
        public void ClearGallery()
        {
            foreach (var exhibit in exhibits)
            {
                if (exhibit != null)
                {
                    exhibit.OnSelected -= HandleExhibitSelected;
                    Destroy(exhibit.gameObject);
                }
            }
            exhibits.Clear();
            exhibitLookup.Clear();

            foreach (var point in teleportPoints)
            {
                if (point != null)
                {
                    Destroy(point);
                }
            }
            teleportPoints.Clear();

            loadedItems.Clear();
        }

        /// <summary>
        /// 특정 전시물 찾기
        /// </summary>
        public ExhibitItem GetExhibit(string itemId)
        {
            exhibitLookup.TryGetValue(itemId, out ExhibitItem exhibit);
            return exhibit;
        }

        /// <summary>
        /// 가장 가까운 전시물 찾기
        /// </summary>
        public ExhibitItem GetNearestExhibit(Vector3 position)
        {
            ExhibitItem nearest = null;
            float minDist = float.MaxValue;

            foreach (var exhibit in exhibits)
            {
                float dist = Vector3.Distance(position, exhibit.transform.position);
                if (dist < minDist)
                {
                    minDist = dist;
                    nearest = exhibit;
                }
            }

            return nearest;
        }

        private int ParseYear(string dateStr)
        {
            if (string.IsNullOrEmpty(dateStr)) return 1970;
            
            if (int.TryParse(dateStr.Substring(0, Math.Min(4, dateStr.Length)), out int year))
            {
                return year;
            }
            return 1970;
        }

        #endregion
    }
}
