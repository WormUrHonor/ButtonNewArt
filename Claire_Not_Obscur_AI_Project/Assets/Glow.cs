using UnityEngine;
using UnityEngine.UI;

public class GlowPulse : MonoBehaviour
{
    public Image image;
    public float speed = 2f;
    public float minAlpha = 0.5f;
    public float maxAlpha = 1f;

    void Update()
    {
        Color c = image.color;
        c.a = Mathf.Lerp(minAlpha, maxAlpha, (Mathf.Sin(Time.time * speed) + 1f)/2f);
        image.color = c;
    }
}
