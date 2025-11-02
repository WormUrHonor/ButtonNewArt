using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class Buttons : MonoBehaviour
{
    

    public void LoadLevelSelecor()
    {
        SceneManager.LoadScene(1);
    }

    public void LoadNN()
    {
        SceneManager.LoadScene(3);
    }

    public void LoadBT()
    {
        SceneManager.LoadScene(4);
    }

    public void LoadPrac()
    {
        SceneManager.LoadScene(2);
    }

    public void LoadMM()
    {
        SceneManager.LoadScene(0);
    }

    public void Reload()
    {
        SceneManager.LoadScene(SceneManager.GetActiveScene().name);
    }

    public void Quit()
    {
        Application.Quit();
    }
}
