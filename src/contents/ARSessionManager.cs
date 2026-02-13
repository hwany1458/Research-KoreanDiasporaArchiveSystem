/*
 * ARSessionManager.cs
 * 
 * AR 세션 관리자
 * AR Foundation을 사용한 AR 기능 초기화 및 관리
 */

using System;
using System.Collections;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using Diaspora.Common;
using Diaspora.API;

namespace Diaspora.AR
{
    /// <summary>
    /// AR 세션 상태
    /// </summary>
    public enum ARSessionState
    {
        None,
        Initializing,
        Ready,
        Tracking,
        Paused,
        Error
    }

    /// <summary>
    /// AR 세션 관리자
    /// </summary>
    public class ARSessionManager : Singleton<ARSessionManager>
    {
        [Header("AR Components")]
        [SerializeField] private ARSession arSession;
        [SerializeField] private ARSessionOrigin arSessionOrigin;
        [SerializeField] private ARCameraManager arCameraManager;
        [SerializeField] private ARTrackedImageManager trackedImageManager;
        [SerializeField] private ARRaycastManager raycastManager;

        [Header("Settings")]
        [SerializeField] private bool autoStartAR = true;
        [SerializeField] private float trackingLostTimeout = 3f;

        [Header("UI References")]
        [SerializeField] private GameObject scanningUI;
        [SerializeField] private GameObject trackingLostUI;

        // 상태
        private ARSessionState currentState = ARSessionState.None;
        private float trackingLostTimer = 0f;
        private bool isTracking = false;

        // 현재 추적 중인 이미지
        private ARTrackedImage currentTrackedImage;
        private ItemData currentItemData;

        // 이벤트
        public event Action<ARSessionState> OnStateChanged;
        public event Action<ARTrackedImage, ItemData> OnImageTracked;
        public event Action OnImageLost;
        public event Action<string> OnError;

        // 프로퍼티
        public ARSessionState CurrentState => currentState;
        public bool IsTracking => isTracking;
        public ARTrackedImage CurrentTrackedImage => currentTrackedImage;
        public ItemData CurrentItemData => currentItemData;
        public Camera ARCamera => arCameraManager?.GetComponent<Camera>();

        protected override void OnSingletonAwake()
        {
            ValidateComponents();
        }

        private void Start()
        {
            if (autoStartAR)
            {
                StartARSession();
            }
        }

        private void OnEnable()
        {
            if (trackedImageManager != null)
            {
                trackedImageManager.trackedImagesChanged += OnTrackedImagesChanged;
            }
        }

        private void OnDisable()
        {
            if (trackedImageManager != null)
            {
                trackedImageManager.trackedImagesChanged -= OnTrackedImagesChanged;
            }
        }

        private void Update()
        {
            // 트래킹 로스트 타이머
            if (isTracking && currentTrackedImage != null)
            {
                if (currentTrackedImage.trackingState != TrackingState.Tracking)
                {
                    trackingLostTimer += Time.deltaTime;
                    if (trackingLostTimer >= trackingLostTimeout)
                    {
                        HandleImageLost();
                    }
                }
                else
                {
                    trackingLostTimer = 0f;
                }
            }
        }

        #region AR 세션 관리

        /// <summary>
        /// AR 세션 시작
        /// </summary>
        public void StartARSession()
        {
            StartCoroutine(InitializeARSession());
        }

        private IEnumerator InitializeARSession()
        {
            SetState(ARSessionState.Initializing);
            ShowScanningUI(true);

            // AR 지원 확인
            if (ARSession.state == UnityEngine.XR.ARFoundation.ARSessionState.None ||
                ARSession.state == UnityEngine.XR.ARFoundation.ARSessionState.CheckingAvailability)
            {
                yield return ARSession.CheckAvailability();
            }

            if (ARSession.state == UnityEngine.XR.ARFoundation.ARSessionState.Unsupported)
            {
                SetState(ARSessionState.Error);
                OnError?.Invoke("이 기기는 AR을 지원하지 않습니다.");
                yield break;
            }

            // AR 세션 활성화
            if (arSession != null)
            {
                arSession.enabled = true;
            }

            // 초기화 대기
            yield return new WaitForSeconds(0.5f);

            SetState(ARSessionState.Ready);
            Debug.Log("[AR] AR 세션이 준비되었습니다.");
        }

        /// <summary>
        /// AR 세션 일시정지
        /// </summary>
        public void PauseARSession()
        {
            if (arSession != null)
            {
                arSession.enabled = false;
            }
            SetState(ARSessionState.Paused);
        }

        /// <summary>
        /// AR 세션 재개
        /// </summary>
        public void ResumeARSession()
        {
            if (arSession != null)
            {
                arSession.enabled = true;
            }
            SetState(ARSessionState.Ready);
        }

        /// <summary>
        /// AR 세션 리셋
        /// </summary>
        public void ResetARSession()
        {
            if (arSession != null)
            {
                arSession.Reset();
            }
            
            currentTrackedImage = null;
            currentItemData = null;
            isTracking = false;
            trackingLostTimer = 0f;
            
            SetState(ARSessionState.Ready);
            ShowScanningUI(true);
        }

        #endregion

        #region 이미지 트래킹

        /// <summary>
        /// 이미지 트래킹 이벤트 핸들러
        /// </summary>
        private void OnTrackedImagesChanged(ARTrackedImagesChangedEventArgs args)
        {
            // 새로 추가된 이미지
            foreach (var trackedImage in args.added)
            {
                HandleImageTracked(trackedImage);
            }

            // 업데이트된 이미지
            foreach (var trackedImage in args.updated)
            {
                if (trackedImage.trackingState == TrackingState.Tracking)
                {
                    if (currentTrackedImage != trackedImage)
                    {
                        HandleImageTracked(trackedImage);
                    }
                }
            }

            // 제거된 이미지
            foreach (var trackedImage in args.removed)
            {
                if (currentTrackedImage == trackedImage)
                {
                    HandleImageLost();
                }
            }
        }

        /// <summary>
        /// 이미지 인식 처리
        /// </summary>
        private void HandleImageTracked(ARTrackedImage trackedImage)
        {
            if (trackedImage.trackingState != TrackingState.Tracking)
                return;

            currentTrackedImage = trackedImage;
            isTracking = true;
            trackingLostTimer = 0f;

            SetState(ARSessionState.Tracking);
            ShowScanningUI(false);
            ShowTrackingLostUI(false);

            // 이미지 이름으로 자료 조회
            string imageName = trackedImage.referenceImage.name;
            Debug.Log($"[AR] 이미지 인식됨: {imageName}");

            // 서버에서 자료 정보 조회
            StartCoroutine(LoadItemData(imageName));
        }

        /// <summary>
        /// 이미지 로스트 처리
        /// </summary>
        private void HandleImageLost()
        {
            Debug.Log("[AR] 이미지 트래킹 로스트");

            isTracking = false;
            currentTrackedImage = null;
            currentItemData = null;
            trackingLostTimer = 0f;

            SetState(ARSessionState.Ready);
            ShowScanningUI(true);
            
            OnImageLost?.Invoke();
        }

        /// <summary>
        /// 자료 데이터 로드
        /// </summary>
        private IEnumerator LoadItemData(string imageName)
        {
            // imageName을 item_id로 사용하거나, 매핑 테이블 사용
            string itemId = imageName;

            yield return APIManager.Instance.GetItem(itemId,
                (ItemData data) =>
                {
                    currentItemData = data;
                    OnImageTracked?.Invoke(currentTrackedImage, data);
                    Debug.Log($"[AR] 자료 로드 완료: {data.title}");
                },
                (string error) =>
                {
                    Debug.LogError($"[AR] 자료 로드 실패: {error}");
                    OnError?.Invoke(error);
                }
            );
        }

        #endregion

        #region 레이캐스트

        /// <summary>
        /// AR 레이캐스트 (화면 터치 위치 → 3D 공간)
        /// </summary>
        public bool TryRaycast(Vector2 screenPosition, out Pose hitPose)
        {
            hitPose = Pose.identity;

            if (raycastManager == null)
                return false;

            var hits = new System.Collections.Generic.List<ARRaycastHit>();
            if (raycastManager.Raycast(screenPosition, hits, TrackableType.AllTypes))
            {
                hitPose = hits[0].pose;
                return true;
            }

            return false;
        }

        /// <summary>
        /// 터치 위치에서 트래킹 이미지 위 좌표 계산
        /// </summary>
        public Vector3 ScreenToImagePosition(Vector2 screenPosition)
        {
            if (currentTrackedImage == null)
                return Vector3.zero;

            Ray ray = ARCamera.ScreenPointToRay(screenPosition);
            Plane imagePlane = new Plane(currentTrackedImage.transform.up, currentTrackedImage.transform.position);

            if (imagePlane.Raycast(ray, out float distance))
            {
                return ray.GetPoint(distance);
            }

            return currentTrackedImage.transform.position;
        }

        #endregion

        #region UI 관리

        private void ShowScanningUI(bool show)
        {
            if (scanningUI != null)
            {
                scanningUI.SetActive(show);
            }
        }

        private void ShowTrackingLostUI(bool show)
        {
            if (trackingLostUI != null)
            {
                trackingLostUI.SetActive(show);
            }
        }

        #endregion

        #region 유틸리티

        private void SetState(ARSessionState newState)
        {
            if (currentState != newState)
            {
                currentState = newState;
                OnStateChanged?.Invoke(newState);
                Debug.Log($"[AR] 상태 변경: {newState}");
            }
        }

        private void ValidateComponents()
        {
            if (arSession == null)
                arSession = FindObjectOfType<ARSession>();
            if (arSessionOrigin == null)
                arSessionOrigin = FindObjectOfType<ARSessionOrigin>();
            if (arCameraManager == null)
                arCameraManager = FindObjectOfType<ARCameraManager>();
            if (trackedImageManager == null)
                trackedImageManager = FindObjectOfType<ARTrackedImageManager>();
            if (raycastManager == null)
                raycastManager = FindObjectOfType<ARRaycastManager>();
        }

        /// <summary>
        /// 트래킹 이미지 크기 (미터)
        /// </summary>
        public Vector2 GetTrackedImageSize()
        {
            if (currentTrackedImage != null)
            {
                return currentTrackedImage.size;
            }
            return Vector2.zero;
        }

        /// <summary>
        /// 트래킹 이미지 월드 위치
        /// </summary>
        public Vector3 GetTrackedImagePosition()
        {
            if (currentTrackedImage != null)
            {
                return currentTrackedImage.transform.position;
            }
            return Vector3.zero;
        }

        /// <summary>
        /// 트래킹 이미지 월드 회전
        /// </summary>
        public Quaternion GetTrackedImageRotation()
        {
            if (currentTrackedImage != null)
            {
                return currentTrackedImage.transform.rotation;
            }
            return Quaternion.identity;
        }

        #endregion
    }
}
